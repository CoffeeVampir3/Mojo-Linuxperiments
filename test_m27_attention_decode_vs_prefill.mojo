from std.memory import UnsafePointer
from std.pathlib import Path
from std.collections import InlineArray

from numa import NumaInfo
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstThreadPool
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.minimax_m27_moe_butterquant_tp import (
    MiniMaxM27Config, MiniMaxM27ButterQuant, MiniMaxShapes,
    MiniMaxM27Topology, PREFILL_CHUNK_SIZE, FWHT_BLK_HIDDEN,
)
from modeling.model_spec import BF16, F32, I8, Shape
from experimental3.common_math import inv_rms_from_sum_sq, rms_reduce_bf16
from experimental3.kernels.dispatch_args import RmsNormFwhtQuantArgs
from experimental3.kernels.dispatch_kernels import int8_gemv
from experimental3.kernels.rmsnorm import rmsnorm_fwht_quant_worker
from experimental3.small_phase_dispatch import run_tp_single_job_phase
from experimental3.profiler import timed_tp_parallel
from minimax.kernels.dispatch_args import MergeQuantArgs
from minimax.kernels.dispatch_kernels import (
    kv_write_dispatch,
    q_prep_batch_dispatch,
    chunked_score_dispatch_multi,
    prefill_attn_dispatch,
    attn_chunk_count,
)
from minimax.kernels.attention import merge_quant_worker
from minimax.kernels.amx_attention import (
    AmxConfigArgs,
    amx_config_kernel,
    amx_prefill_config_kernel,
)
from kernels.kernel_ops import PoolFence, embed_lookup, MAX_POOL_CAPACITY
from kernels.reductions import ring_broadcast


comptime C = MiniMaxM27Config
comptime TP = 4
comptime TOKENIZER_PATH = "checkpoints/Minimax-M2.7/tokenizer.json"
comptime MODEL_DIR = "quantized_models"
comptime HEAD_DIM = C.HEAD_DIM
comptime HPG = C.HPG


def abs_f32(x: Float32) -> Float32:
    if x < Float32(0):
        return -x
    return x


def max_f32(a: Float32, b: Float32) -> Float32:
    if a > b:
        return a
    return b


def set_decode_amx[P: BurstThreadPool, //, tp: Int](
    mut model: MiniMaxM27ButterQuant[tp, P],
):
    for rank in range(tp):
        var args = InlineArray[AmxConfigArgs, MAX_POOL_CAPACITY](
            fill=AmxConfigArgs())
        var cap = model.main_pools[rank].get_capacity()
        model.main_pools[rank].dispatch[
            AmxConfigArgs, amx_config_kernel[C.HPG]](
            UnsafePointer(to=args[0]), cap)
        model.main_pools[rank].join()


def set_prefill_amx[P: BurstThreadPool, //, tp: Int](
    mut model: MiniMaxM27ButterQuant[tp, P],
):
    for rank in range(tp):
        var args = InlineArray[AmxConfigArgs, MAX_POOL_CAPACITY](
            fill=AmxConfigArgs())
        var cap = model.main_pools[rank].get_capacity()
        model.main_pools[rank].dispatch[
            AmxConfigArgs, amx_prefill_config_kernel](
            UnsafePointer(to=args[0]), cap)
        model.main_pools[rank].join()


def run_attention_comparison[P: BurstThreadPool, //, tp: Int](
    mut model: MiniMaxM27ButterQuant[tp, P],
    read token_ids: List[Int],
):
    comptime S = MiniMaxShapes[tp]
    comptime EPS = Float32(C.RMS_NORM_EPS)
    comptime Q_LOCAL = S.Q_LOCAL
    comptime KV_LOCAL = S.KV_LOCAL
    comptime QKV_LOCAL = S.QKV_LOCAL
    comptime KV_PER_RANK = S.NUM_KV_HEADS_LOCAL
    comptime HEADS_PER_RANK = S.NUM_HEADS_LOCAL
    comptime ROPE_HALF = C.ROPE_DIM // 2
    comptime XShape = Shape[C.MAX_SEQ_LEN, C.HIDDEN]
    comptime Q_PREP_ELEMS = KV_PER_RANK * HPG * HEAD_DIM
    comptime Q_PREP_META = KV_PER_RANK * HPG
    comptime PARTIAL_F32S = (
        KV_PER_RANK * C.MAX_ATTN_CHUNKS * HPG * (2 + HEAD_DIM))

    var seq_len = len(token_ids)
    if seq_len < 2:
        print("need at least two prompt tokens")
        return
    if seq_len > PREFILL_CHUNK_SIZE:
        print(
            "prompt has", seq_len, "tokens but this test compares one prefill chunk of",
            PREFILL_CHUNK_SIZE,
        )
        return

    var topos = model.topos
    var host = topos[0]
    var tp_ptr = model.token_buffer()
    for i in range(seq_len):
        tp_ptr[i] = Scalar[DType.int32](token_ids[i])

    print("m27 attention decode-vs-prefill comparison")
    print("tokens=", seq_len, " tp=", tp, " layer=0")

    # Production setup through the same model kernels used before attention.
    embed_lookup(
        host.host.embed.bound(host.arena.base),
        Int(tp_ptr),
        host.activations.x_main.bound_dyn(host.arena.base, seq_len),
        model.main_pools[0],
    ).join()
    ring_broadcast[BF16, XShape, tp](
        host.activations.x_main.addr(host.arena.base),
        model.x_main_ptrs(),
        seq_len,
        model.main_pools,
    )

    var act_scale_lease = model.scratch.borrow[Float32, PREFILL_CHUNK_SIZE]()
    comptime layer_idx = 0
    var qkv_lease = model.scratch.borrow[
        Scalar[DType.bfloat16], PREFILL_CHUNK_SIZE * QKV_LOCAL]()
    var attn_i8_lease = model.scratch.borrow[
        Scalar[DType.int8], PREFILL_CHUNK_SIZE * C.HIDDEN]()
    var attn_work_lease = model.scratch.borrow[Float32, C.HIDDEN]()

    var norm_jobs = InlineArray[RmsNormFwhtQuantArgs, tp](uninitialized=True)
    for r in range(tp):
        var topo = topos[r]
        var lb = topo.layers.base(topo.arena.base, layer_idx)
        var sb = topo.arena.scratch_base()
        norm_jobs[r] = RmsNormFwhtQuantArgs(
            topo.activations.x_main.bound_dyn(topo.arena.base, seq_len)
                .as_ptr[DType.bfloat16](),
            topo.layers.proto.body.input_norm_sqrt.bound(lb)
                .as_ptr[DType.bfloat16](),
            attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](
                sb, seq_len).as_ptr[DType.int8](),
            attn_work_lease.view[F32, Shape[1, C.HIDDEN]](
                sb, 1).as_ptr[DType.float32](),
            act_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](
                sb, seq_len).as_ptr[DType.float32](),
            EPS, 0, seq_len)
    _ = run_tp_single_job_phase[
        tp,
        rmsnorm_fwht_quant_worker[C.HIDDEN, FWHT_BLK_HIDDEN, True, False],
    ](UnsafePointer(to=norm_jobs[0]), model.main_pools)

    @parameter
    def do_qkv[
        rank: Int, origin: MutOrigin,
    ](
        topo: MiniMaxM27Topology[tp],
        ref [origin] pool: P,
    ) -> PoolFence[P, origin]:
        var lb = topo.layers.base(topo.arena.base, layer_idx)
        var sb = topo.arena.scratch_base()
        return int8_gemv[QKV_LOCAL, C.HIDDEN](
            attn_i8_lease.view[I8, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](
                sb, seq_len),
            topo.layers.proto.attn.qkv_proj.bound(lb),
            topo.layers.proto.attn.qkv_colsum.bound(lb),
            topo.layers.proto.attn.qkv_proj_sc.bound(lb),
            qkv_lease.view[BF16, Shape[C.MAX_SEQ_LEN, QKV_LOCAL]](
                sb, seq_len),
            act_scale_lease.view[F32, Shape[C.MAX_SEQ_LEN, 1]](
                sb, seq_len),
            pool)
    _ = timed_tp_parallel[tp, do_qkv](topos, model.main_pools)

    attn_work_lease^.release()
    attn_i8_lease^.release()

    var inv_rms_q_arr = InlineArray[Float32, PREFILL_CHUNK_SIZE](
        fill=Float32(0))
    var inv_rms_k_arr = InlineArray[Float32, PREFILL_CHUNK_SIZE](
        fill=Float32(0))
    for row in range(seq_len):
        var q_ss = Float32(0)
        var k_ss = Float32(0)
        for r in range(tp):
            var qkv_ptr = qkv_lease.as_ptr[Scalar[DType.bfloat16]](
                topos[r].arena.scratch_base(), row * QKV_LOCAL)
            q_ss += rms_reduce_bf16[Q_LOCAL](qkv_ptr)
            k_ss += rms_reduce_bf16[KV_LOCAL](qkv_ptr + Q_LOCAL)
        inv_rms_q_arr[row] = inv_rms_from_sum_sq(q_ss, C.Q_DIM, EPS)
        inv_rms_k_arr[row] = inv_rms_from_sum_sq(k_ss, C.KV_DIM, EPS)

    @parameter
    def do_kv_write[
        rank: Int, origin: MutOrigin,
    ](
        topo: MiniMaxM27Topology[tp],
        ref [origin] pool: P,
    ) -> PoolFence[P, origin]:
        var lb = topo.layers.base(topo.arena.base, layer_idx)
        var sb = topo.arena.scratch_base()
        return kv_write_dispatch[
            HEAD_DIM, C.ROPE_DIM, C.ROPE_PAIR_STRIDE,
            C.MAX_SEQ_LEN, KV_PER_RANK, QKV_LOCAL, Q_LOCAL, ROPE_HALF,
        ](
            qkv_lease.as_ptr[Scalar[DType.bfloat16]](sb),
            topo.layers.proto.attn.k_norm.bound(lb).as_ptr[DType.bfloat16](),
            topo.rope.cos.bound_row(topo.arena.base, 0).as_ptr[DType.float32](),
            topo.rope.sin.bound_row(topo.arena.base, 0).as_ptr[DType.float32](),
            UnsafePointer(to=inv_rms_k_arr[0]).bitcast[Float32]().as_any_origin(),
            topo.arena.base + topo.kv_cache_off + layer_idx * topo.kv_cache_stride,
            0, seq_len, pool)
    _ = timed_tp_parallel[tp, do_kv_write](topos, model.main_pools)

    var q_i8_lease = model.scratch.borrow[
        Scalar[DType.int8], PREFILL_CHUNK_SIZE * Q_PREP_ELEMS]()
    var qi_biases_lease = model.scratch.borrow[
        Float32, PREFILL_CHUNK_SIZE * Q_PREP_META]()
    var q_scales_lease = model.scratch.borrow[
        Float32, PREFILL_CHUNK_SIZE * Q_PREP_META]()

    for r in range(tp):
        var topo = topos[r]
        var lb = topo.layers.base(topo.arena.base, layer_idx)
        var sb = topo.arena.scratch_base()
        q_prep_batch_dispatch[
            HEAD_DIM, C.ROPE_DIM, C.ROPE_PAIR_STRIDE,
            HPG, KV_PER_RANK, QKV_LOCAL, ROPE_HALF,
        ](
            qkv_lease.as_ptr[Scalar[DType.bfloat16]](sb),
            topo.layers.proto.attn.q_norm.bound(lb).as_ptr[DType.bfloat16](),
            topo.rope.cos.bound_row(topo.arena.base, 0).as_ptr[DType.float32](),
            topo.rope.sin.bound_row(topo.arena.base, 0).as_ptr[DType.float32](),
            UnsafePointer(to=inv_rms_q_arr[0]).bitcast[Float32]().as_any_origin(),
            q_i8_lease.as_ptr[Scalar[DType.int8]](sb),
            qi_biases_lease.as_ptr[Float32](sb),
            q_scales_lease.as_ptr[Float32](sb),
            0, seq_len)

    # Decode reference: use the actual decode attention dispatcher and merge
    # worker, one prompt position at a time, after re-layouting the batch Q prep
    # result into the decode Q layout expected by amx_chunked_attn_kernel.
    var decode_q_lease = model.scratch.borrow[
        Scalar[DType.int8], Q_PREP_ELEMS]()
    var decode_bias_lease = model.scratch.borrow[Float32, Q_PREP_META]()
    var decode_scale_lease = model.scratch.borrow[Float32, Q_PREP_META]()
    var partial_lease = model.scratch.borrow[Float32, PARTIAL_F32S]()
    var ref_qi_lease = model.scratch.borrow[
        Scalar[DType.int8], PREFILL_CHUNK_SIZE * Q_LOCAL]()
    var ref_sc_lease = model.scratch.borrow[
        Float32, PREFILL_CHUNK_SIZE * HEADS_PER_RANK]()
    var prefill_qi_lease = model.scratch.borrow[
        Scalar[DType.int8], PREFILL_CHUNK_SIZE * Q_LOCAL]()
    var prefill_sc_lease = model.scratch.borrow[
        Float32, PREFILL_CHUNK_SIZE * HEADS_PER_RANK]()

    set_decode_amx[tp](model)
    for pos in range(seq_len):
        for r in range(tp):
            var sb = topos[r].arena.scratch_base()
            var batch_q = q_i8_lease.as_ptr[Scalar[DType.int8]](sb)
            var batch_bias = qi_biases_lease.as_ptr[Float32](sb)
            var batch_scale = q_scales_lease.as_ptr[Float32](sb)
            var dec_q = decode_q_lease.as_ptr[Scalar[DType.int8]](sb)
            var dec_bias = decode_bias_lease.as_ptr[Float32](sb)
            var dec_scale = decode_scale_lease.as_ptr[Float32](sb)
            for kv in range(KV_PER_RANK):
                for qh in range(HPG):
                    var src_q = batch_q + (kv * HPG + qh) * seq_len * HEAD_DIM + pos * HEAD_DIM
                    var dst_q = dec_q + kv * HPG * HEAD_DIM + qh * HEAD_DIM
                    comptime for chunk in range(HEAD_DIM // 64):
                        (dst_q + chunk * 64).store(
                            (src_q + chunk * 64).load[width=64]())
                    dec_bias[kv * HPG + qh] = batch_bias[
                        (kv * HPG + qh) * seq_len + pos]
                    dec_scale[kv * HPG + qh] = batch_scale[
                        (kv * HPG + qh) * seq_len + pos]

        var context_len = pos + 1

        @parameter
        def do_decode_attention[
            rank: Int, origin: MutOrigin,
        ](
            topo: MiniMaxM27Topology[tp],
            ref [origin] pool: P,
        ) -> PoolFence[P, origin]:
            var sb = topo.arena.scratch_base()
            return chunked_score_dispatch_multi[
                HEAD_DIM, HPG, C.MAX_SEQ_LEN, KV_PER_RANK,
                C.MAX_ATTN_CHUNKS,
            ](
                Int(decode_q_lease.as_ptr[Scalar[DType.int8]](sb)),
                Int(decode_bias_lease.as_ptr[Float32](sb)),
                Int(decode_scale_lease.as_ptr[Float32](sb)),
                topo.arena.base + topo.kv_cache_off + layer_idx * topo.kv_cache_stride,
                0, KV_PER_RANK,
                context_len, Int(pool.get_capacity()),
                Int(partial_lease.as_ptr[Float32](sb)),
                pool)
        _ = timed_tp_parallel[tp, do_decode_attention](topos, model.main_pools)

        var merge_jobs = InlineArray[MergeQuantArgs, tp](uninitialized=True)
        for r in range(tp):
            var sb = topos[r].arena.scratch_base()
            merge_jobs[r] = MergeQuantArgs(
                partial_lease.view[F32, Shape[1, PARTIAL_F32S]](
                    sb, 1).as_ptr[DType.float32](),
                attn_chunk_count(context_len, C.MAX_ATTN_CHUNKS),
                ref_qi_lease.view[
                    I8, Shape[1, KV_PER_RANK * HPG * HEAD_DIM]](
                    sb, 1, element_offset=pos * Q_LOCAL).as_ptr[DType.int8](),
                ref_sc_lease.view[
                    F32, Shape[1, KV_PER_RANK * HPG]](
                    sb, 1, element_offset=pos * HEADS_PER_RANK)
                    .as_ptr[DType.float32](),
            )
        _ = run_tp_single_job_phase[
            tp,
            merge_quant_worker[HEAD_DIM, HPG, C.MAX_ATTN_CHUNKS, KV_PER_RANK],
        ](UnsafePointer(to=merge_jobs[0]), model.main_pools)

    # Prefill candidate: use the actual prefill attention dispatcher that the
    # model selects when seq_len > 1.
    set_prefill_amx[tp](model)

    @parameter
    def do_prefill_attention[
        rank: Int, origin: MutOrigin,
    ](
        topo: MiniMaxM27Topology[tp],
        ref [origin] pool: P,
    ) -> PoolFence[P, origin]:
        var sb = topo.arena.scratch_base()
        return prefill_attn_dispatch[
            HEAD_DIM, HPG, C.MAX_SEQ_LEN, KV_PER_RANK,
            C.NUM_HEADS, Q_LOCAL, HEADS_PER_RANK,
        ](
            q_i8_lease.as_ptr[Scalar[DType.int8]](sb),
            qi_biases_lease.as_ptr[Float32](sb),
            q_scales_lease.as_ptr[Float32](sb),
            topo.arena.base + topo.kv_cache_off + layer_idx * topo.kv_cache_stride,
            0, seq_len,
            prefill_qi_lease.as_ptr[Scalar[DType.int8]](sb),
            prefill_sc_lease.as_ptr[Float32](sb),
            pool)
    _ = timed_tp_parallel[tp, do_prefill_attention](topos, model.main_pools)
    set_decode_amx[tp](model)

    # Numerical comparison on the exact pre-O-projection representation consumed
    # by the model: i8 attention rows plus one dynamic scale per attention head.
    var total_i8_elems = 0
    var total_i8_abs_diff = 0
    var max_i8_diff = 0
    var max_i8_rank = 0
    var max_i8_row = 0
    var max_i8_col = 0

    var total_deq_abs_diff = Float64(0)
    var max_deq_diff = Float32(0)
    var max_deq_rank = 0
    var max_deq_row = 0
    var max_deq_col = 0

    var total_scale_abs_diff = Float64(0)
    var max_scale_abs_diff = Float32(0)
    var max_scale_rel_diff = Float32(0)
    var max_scale_rank = 0
    var max_scale_row = 0
    var max_scale_head = 0

    print("")
    print("per-row summary:")
    print("rank row | i8_max i8_mean deq_max deq_mean scale_max scale_rel")

    for rank in range(tp):
        var sb = topos[rank].arena.scratch_base()
        var ref_qi = ref_qi_lease.as_ptr[Scalar[DType.int8]](sb)
        var pf_qi = prefill_qi_lease.as_ptr[Scalar[DType.int8]](sb)
        var ref_sc = ref_sc_lease.as_ptr[Float32](sb)
        var pf_sc = prefill_sc_lease.as_ptr[Float32](sb)

        for row in range(seq_len):
            var row_i8_abs = 0
            var row_i8_max = 0
            var row_deq_abs = Float64(0)
            var row_deq_max = Float32(0)
            var row_scale_max = Float32(0)
            var row_scale_rel = Float32(0)

            for h in range(HEADS_PER_RANK):
                var rs = ref_sc[row * HEADS_PER_RANK + h]
                var ps = pf_sc[row * HEADS_PER_RANK + h]
                var sd = abs_f32(rs - ps)
                var denom = max_f32(abs_f32(rs), Float32(1e-8))
                var rel = sd / denom
                total_scale_abs_diff += Float64(sd)
                if sd > row_scale_max:
                    row_scale_max = sd
                if rel > row_scale_rel:
                    row_scale_rel = rel
                if sd > max_scale_abs_diff:
                    max_scale_abs_diff = sd
                    max_scale_rank = rank
                    max_scale_row = row
                    max_scale_head = h
                if rel > max_scale_rel_diff:
                    max_scale_rel_diff = rel

            for col in range(Q_LOCAL):
                var idx = row * Q_LOCAL + col
                var rv = Int(ref_qi[idx])
                var pv = Int(pf_qi[idx])
                var d = rv - pv
                if d < 0:
                    d = -d
                row_i8_abs += d
                total_i8_abs_diff += d
                total_i8_elems += 1
                if d > row_i8_max:
                    row_i8_max = d
                if d > max_i8_diff:
                    max_i8_diff = d
                    max_i8_rank = rank
                    max_i8_row = row
                    max_i8_col = col

                var head = col // HEAD_DIM
                var r_deq = (
                    Float32(rv) * ref_sc[row * HEADS_PER_RANK + head]
                    / Float32(127))
                var p_deq = (
                    Float32(pv) * pf_sc[row * HEADS_PER_RANK + head]
                    / Float32(127))
                var dd = abs_f32(r_deq - p_deq)
                row_deq_abs += Float64(dd)
                total_deq_abs_diff += Float64(dd)
                if dd > row_deq_max:
                    row_deq_max = dd
                if dd > max_deq_diff:
                    max_deq_diff = dd
                    max_deq_rank = rank
                    max_deq_row = row
                    max_deq_col = col

            var row_i8_mean = Float64(row_i8_abs) / Float64(Q_LOCAL)
            var row_deq_mean = row_deq_abs / Float64(Q_LOCAL)

            if row < 8 or row == seq_len - 1 or row_i8_max > 2 or row_deq_max > Float32(0.02) or row_scale_rel > Float32(0.001):
                print(
                    rank, row, "|",
                    row_i8_max, row_i8_mean,
                    row_deq_max, row_deq_mean,
                    row_scale_max, row_scale_rel,
                )

    print("")
    print("global summary:")
    print(
        "i8 mean_abs=", Float64(total_i8_abs_diff) / Float64(total_i8_elems),
        " max=", max_i8_diff,
        " at rank,row,col=", max_i8_rank, max_i8_row, max_i8_col,
    )
    print(
        "dequant mean_abs=", total_deq_abs_diff / Float64(total_i8_elems),
        " max=", max_deq_diff,
        " at rank,row,col=", max_deq_rank, max_deq_row, max_deq_col,
    )
    print(
        "scale mean_abs=",
        total_scale_abs_diff / Float64(tp * seq_len * HEADS_PER_RANK),
        " max_abs=", max_scale_abs_diff,
        " max_rel=", max_scale_rel_diff,
        " at rank,row,head=", max_scale_rank, max_scale_row, max_scale_head,
    )

    prefill_sc_lease^.release()
    prefill_qi_lease^.release()
    ref_sc_lease^.release()
    ref_qi_lease^.release()
    partial_lease^.release()
    decode_scale_lease^.release()
    decode_bias_lease^.release()
    decode_q_lease^.release()
    q_scales_lease^.release()
    qi_biases_lease^.release()
    q_i8_lease^.release()
    qkv_lease^.release()
    act_scale_lease^.release()


def main():
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("failed to load tokenizer")
        return
    var tok = tok_opt.take()

    var prompt = "]~!b[]~b]system\nYou are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.[e~[\n]~b]user\nThe capital of France[e~[\n]~b]ai\n<think>\n"
    var token_ids = tok.encode(prompt)

    var numa = NumaInfo()
    var numa_topo = numa.plan_topology(TP)

    if numa.has_isolation():
        var pools = HeapMoveArray[IsolatedBurstPool[]](TP)
        for rank in range(TP):
            pools.push(IsolatedBurstPool[].for_topology(numa, numa_topo[rank]))
        var model_opt = MiniMaxM27ButterQuant[TP, IsolatedBurstPool[]].load(
            Path(MODEL_DIR), numa_topo, pools^)
        if not model_opt:
            return
        var model = model_opt.take()
        run_attention_comparison[TP](model, token_ids)
    else:
        var pools = HeapMoveArray[BurstPool[]](TP)
        for rank in range(TP):
            pools.push(BurstPool[].for_topology(numa, numa_topo[rank]))
        var model_opt = MiniMaxM27ButterQuant[TP, BurstPool[]].load(
            Path(MODEL_DIR), numa_topo, pools^)
        if not model_opt:
            return
        var model = model_opt.take()
        run_attention_comparison[TP](model, token_ids)
