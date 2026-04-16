"""Correctness test for gemv_row_blocked_wa.

Tests the fwht.mojo-style SIMD pattern for blocked GEMV:
  width = min(VNNI_TILE_N, simd_width_of[int32]())
  regs  = VNNI_TILE_N // width

Validates against scalar reference at multiple block sizes, and against the
standard gemv_row_blocked at blk=64 (exact match required).
"""

from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.collections import InlineArray

from threading import BurstPool
from kernels.vnni import pack_vnni, VNNI_N_STEP, VNNI_K_STEP, VNNI_TILE_N, VNNI_BLK, compute_n_block
from experimental3.kernels.int8_gemv import dot
from experimental3.kernels.fwht import fwht_block
from experimental3.kernels.quantize import absmax_quantize_i8
from experimental3.kernels.gelu_tanh_fwht_quantize import gelu_tanh_f32
from experimental3.moe import gemv_row_blocked
from experimental3.kernels.dense_ffn_workaround import gemv_row_blocked_wa, fused_gu_gelu_tanh_wa


# --- test data ---


struct TestData[N: Int, K: Int]:
    var act: List[Scalar[DType.int8]]
    var weight: List[Scalar[DType.int8]]
    var packed: List[UInt8]
    var wscale: List[Float32]

    def __init__(out self):
        self.act = List[Scalar[DType.int8]](capacity=Self.K)
        for i in range(Self.K):
            self.act.append(Scalar[DType.int8]((i * 7 + 13) % 251 - 125))

        self.weight = List[Scalar[DType.int8]](capacity=Self.N * Self.K)
        for i in range(Self.N * Self.K):
            self.weight.append(Scalar[DType.int8]((i * 11 + 3) % 251 - 125))

        self.wscale = List[Float32](capacity=Self.N)
        for i in range(Self.N):
            self.wscale.append(0.01 + Float32(i % 7) * 0.005)

        self.packed = List[UInt8](capacity=Self.N * Self.K)
        for _ in range(Self.N * Self.K):
            self.packed.append(0)
        var tmp = List[UInt8](capacity=Self.N * Self.K)
        var w_u8 = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(self.weight.unsafe_ptr()))
        for i in range(Self.N * Self.K):
            tmp.append(w_u8[i])
        pack_vnni(
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(tmp.unsafe_ptr())),
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(self.packed.unsafe_ptr())),
            Self.N, Self.K)

    def act_ptr(ref self) -> UnsafePointer[Scalar[DType.int8], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(self.act.unsafe_ptr()))

    def weight_ptr(ref self) -> UnsafePointer[Scalar[DType.int8], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(self.weight.unsafe_ptr()))

    def packed_ptr(ref self) -> UnsafePointer[UInt8, MutAnyOrigin]:
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(self.packed.unsafe_ptr()))

    def wscale_ptr(ref self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=Int(self.wscale.unsafe_ptr()))


def make_blk_scales(num_blocks: Int) -> List[Float32]:
    var out = List[Float32](capacity=num_blocks)
    for i in range(num_blocks):
        out.append(0.5 + Float32(i) * 0.1)
    return out^


def compute_colsums[N: Int, K: Int](
    weight: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    blk: Int,
) -> List[Float32]:
    var num_blocks = K // blk
    var out = List[Float32](capacity=num_blocks * N)
    for _ in range(num_blocks * N):
        out.append(0)
    var ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(out.unsafe_ptr()))
    for b in range(num_blocks):
        for n in range(N):
            var s = Float32(0)
            for ki in range(blk):
                s += Float32(Int(weight[n * K + b * blk + ki]))
            ptr[b * N + n] = s
    return out^


def compute_row_colsums[ROWS: Int, COLS: Int](
    weight: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
) -> List[Float32]:
    var out = List[Float32](capacity=ROWS)
    for _ in range(ROWS):
        out.append(0)
    var ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(out.unsafe_ptr()))
    for r in range(ROWS):
        var s = Float32(0)
        for c in range(COLS):
            s += Float32(Int(weight[r * COLS + c]))
        ptr[r] = s
    return out^


# --- scalar reference ---


def scalar_gemv_blocked[N: Int, K: Int](
    act: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    blk_scales: UnsafePointer[Float32, MutAnyOrigin],
    wscales: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    blk: Int,
):
    var num_blocks = K // blk
    for n in range(N):
        var acc = Float32(0)
        for b in range(num_blocks):
            var blk_acc = Float32(0)
            for ki in range(blk):
                var k = b * blk + ki
                blk_acc += Float32(Int(weight[n * K + k])) * Float32(Int(act[k]))
            acc += blk_acc * (blk_scales[b] / 127.0)
        dst[n] = acc * wscales[n]


def scalar_rowwise_corrected[N: Int, K: Int](
    act: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    act_scale: Float32,
    weight: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wscales: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    var dq = act_scale / 127.0
    for n in range(N):
        var acc = Float32(0)
        for k in range(K):
            acc += Float32(Int(weight[n * K + k])) * Float32(Int(act[k]))
        dst[n] = acc * dq * wscales[n]


struct DenseWaData[INTER: Int, ACTUAL: Int, HIDDEN: Int, OUT: Int]:
    var act: List[Scalar[DType.int8]]
    var act_scale: List[Float32]
    var fused_weight: List[Scalar[DType.int8]]
    var fused_packed: List[UInt8]
    var fused_wscale: List[Float32]
    var down_weight: List[Scalar[DType.int8]]
    var down_packed: List[UInt8]
    var down_wscale: List[Float32]

    def __init__(out self):
        self.act = List[Scalar[DType.int8]](capacity=Self.HIDDEN)
        for i in range(Self.HIDDEN):
            self.act.append(Scalar[DType.int8]((i * 17 + 5) % 251 - 125))

        self.act_scale = List[Float32](capacity=1)
        self.act_scale.append(1.75)

        self.fused_weight = List[Scalar[DType.int8]](capacity=2 * Self.INTER * Self.HIDDEN)
        for r in range(2 * Self.INTER):
            var local_r = r % Self.INTER
            for c in range(Self.HIDDEN):
                var v = ((r * 13 + c * 7 + 9) % 251) - 125
                if local_r >= Self.ACTUAL:
                    v = 0
                self.fused_weight.append(Scalar[DType.int8](v))

        self.fused_wscale = List[Float32](capacity=2 * Self.INTER)
        for r in range(2 * Self.INTER):
            self.fused_wscale.append(0.02 + Float32(r % 11) * 0.003)

        self.fused_packed = List[UInt8](capacity=2 * Self.INTER * Self.HIDDEN)
        for _ in range(2 * Self.INTER * Self.HIDDEN):
            self.fused_packed.append(0)
        var fused_tmp = List[UInt8](capacity=2 * Self.INTER * Self.HIDDEN)
        var fused_u8 = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(self.fused_weight.unsafe_ptr()))
        for i in range(2 * Self.INTER * Self.HIDDEN):
            fused_tmp.append(fused_u8[i])
        pack_vnni(
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(fused_tmp.unsafe_ptr())),
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(self.fused_packed.unsafe_ptr())),
            2 * Self.INTER, Self.HIDDEN)

        self.down_weight = List[Scalar[DType.int8]](capacity=Self.OUT * Self.INTER)
        for r in range(Self.OUT):
            for c in range(Self.INTER):
                var v = ((r * 19 + c * 5 + 21) % 251) - 125
                self.down_weight.append(Scalar[DType.int8](v))

        self.down_wscale = List[Float32](capacity=Self.OUT)
        for r in range(Self.OUT):
            self.down_wscale.append(0.015 + Float32(r % 13) * 0.002)

        self.down_packed = List[UInt8](capacity=Self.OUT * Self.INTER)
        for _ in range(Self.OUT * Self.INTER):
            self.down_packed.append(0)
        var down_tmp = List[UInt8](capacity=Self.OUT * Self.INTER)
        var down_u8 = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(self.down_weight.unsafe_ptr()))
        for i in range(Self.OUT * Self.INTER):
            down_tmp.append(down_u8[i])
        pack_vnni(
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(down_tmp.unsafe_ptr())),
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(self.down_packed.unsafe_ptr())),
            Self.OUT, Self.INTER)

    def act_ptr(ref self) -> UnsafePointer[Scalar[DType.int8], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(self.act.unsafe_ptr()))

    def act_scale_ptr(ref self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=Int(self.act_scale.unsafe_ptr()))

    def fused_weight_ptr(ref self) -> UnsafePointer[Scalar[DType.int8], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(self.fused_weight.unsafe_ptr()))

    def fused_packed_ptr(ref self) -> UnsafePointer[UInt8, MutAnyOrigin]:
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(self.fused_packed.unsafe_ptr()))

    def fused_wscale_ptr(ref self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=Int(self.fused_wscale.unsafe_ptr()))

    def down_weight_ptr(ref self) -> UnsafePointer[Scalar[DType.int8], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(self.down_weight.unsafe_ptr()))

    def down_packed_ptr(ref self) -> UnsafePointer[UInt8, MutAnyOrigin]:
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(self.down_packed.unsafe_ptr()))

    def down_wscale_ptr(ref self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=Int(self.down_wscale.unsafe_ptr()))


def max_scale_diff(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> Float32:
    return max_abs_diff(a, b, n)


def count_i8_mismatch(
    a: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    n: Int,
) -> Int:
    var mismatches = 0
    for i in range(n):
        if a[i] != b[i]:
            mismatches += 1
    return mismatches


def run_phase1_ref[INTER: Int, HIDDEN: Int, BLK: Int](
    td: DenseWaData[INTER, _, HIDDEN, _],
    qi_out: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    blk_scale: UnsafePointer[Float32, MutAnyOrigin],
):
    var gate = List[Float32](capacity=INTER)
    var up = List[Float32](capacity=INTER)
    var inter = List[Float32](capacity=INTER)
    for _ in range(INTER):
        gate.append(0)
        up.append(0)
        inter.append(0)

    var gate_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(gate.unsafe_ptr()))
    var up_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(up.unsafe_ptr()))
    var inter_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(inter.unsafe_ptr()))

    scalar_rowwise_corrected[INTER, HIDDEN](
        td.act_ptr(), td.act_scale[0], td.fused_weight_ptr(), td.fused_wscale_ptr(), gate_ptr)
    scalar_rowwise_corrected[INTER, HIDDEN](
        td.act_ptr(), td.act_scale[0],
        td.fused_weight_ptr() + INTER * HIDDEN,
        td.fused_wscale_ptr() + INTER,
        up_ptr)

    comptime width = simd_width_of[DType.float32]()
    var k = 0
    while k + width <= INTER:
        var g = (gate_ptr + k).load[width=width]()
        var u = (up_ptr + k).load[width=width]()
        (inter_ptr + k).store(gelu_tanh_f32[width](g) * u)
        k += width

    for b in range(INTER // BLK):
        var off = b * BLK
        fwht_block[BLK](inter_ptr + off)
        blk_scale[b] = absmax_quantize_i8[BLK](inter_ptr + off, qi_out + off)


def run_e2e_test[INTER: Int, ACTUAL: Int, HIDDEN: Int, OUT: Int, BLK: Int](label: String):
    print("--- " + label + " ---")
    var td = DenseWaData[INTER, ACTUAL, HIDDEN, OUT]()
    var fused_colsums = compute_row_colsums[2 * INTER, HIDDEN](td.fused_weight_ptr())
    var down_colsums = compute_colsums[OUT, INTER](td.down_weight_ptr(), BLK)
    var fused_cs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(fused_colsums.unsafe_ptr()))
    var down_cs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(down_colsums.unsafe_ptr()))

    var qi_ref = List[Scalar[DType.int8]](capacity=INTER)
    var qi_kernel = List[Scalar[DType.int8]](capacity=INTER)
    for _ in range(INTER):
        qi_ref.append(0)
        qi_kernel.append(0)
    var sc_ref = List[Float32](capacity=INTER // BLK)
    var sc_kernel = List[Float32](capacity=INTER // BLK)
    for _ in range(INTER // BLK):
        sc_ref.append(0)
        sc_kernel.append(0)

    var qi_ref_ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=Int(qi_ref.unsafe_ptr()))
    var qi_kernel_ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](unsafe_from_address=Int(qi_kernel.unsafe_ptr()))
    var sc_ref_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(sc_ref.unsafe_ptr()))
    var sc_kernel_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(sc_kernel.unsafe_ptr()))

    run_phase1_ref[INTER, HIDDEN, BLK](td, qi_ref_ptr, sc_ref_ptr)

    var pool = BurstPool[](4)
    if not pool:
        print("FAIL could not create BurstPool")
        return
    fused_gu_gelu_tanh_wa[INTER, HIDDEN, BLK](
        td.act_ptr(), td.act_scale_ptr(),
        td.fused_packed_ptr(), td.fused_wscale_ptr(), fused_cs,
        qi_kernel_ptr, sc_kernel_ptr, 1, pool,
    ).join()

    var qi_mismatches = count_i8_mismatch(qi_ref_ptr, qi_kernel_ptr, INTER)
    var sc_diff = max_scale_diff(sc_ref_ptr, sc_kernel_ptr, INTER // BLK)

    var out_ref = List[Float32](capacity=OUT)
    var out_kernel = List[Float32](capacity=OUT)
    for _ in range(OUT):
        out_ref.append(0)
        out_kernel.append(0)
    var out_ref_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(out_ref.unsafe_ptr()))
    var out_kernel_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(out_kernel.unsafe_ptr()))

    scalar_gemv_blocked[OUT, INTER](
        qi_ref_ptr, td.down_weight_ptr(), sc_ref_ptr, td.down_wscale_ptr(), out_ref_ptr, BLK)
    gemv_row_blocked_wa[OUT, INTER, BLK](
        qi_kernel_ptr, td.down_packed_ptr(), sc_kernel_ptr, td.down_wscale_ptr(), down_cs, out_kernel_ptr)

    var out_diff = max_abs_diff(out_ref_ptr, out_kernel_ptr, OUT)
    if qi_mismatches == 0 and sc_diff == 0 and out_diff == 0:
        print("PASS (phase1 + phase2 exact)")
    else:
        print("FAIL qi mismatches:", qi_mismatches, " scale diff:", sc_diff, " out diff:", out_diff)
        for i in range(min(4, OUT)):
            print("  [", i, "] ref:", out_ref_ptr[i], " kernel:", out_kernel_ptr[i])


# --- new kernel under test (fwht.mojo-style SIMD pattern) ---


def gemv_tile_width[T: DType, tile: Int]() -> Int:
    """Same pattern as fwht_width: use the tile size or hardware width, whichever is smaller."""
    comptime hw = simd_width_of[T]()
    comptime if tile <= hw:
        return tile
    else:
        return hw


def gemv_row_blocked_new[N: Int, K: Int, fwht_blk: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[UInt8, MutAnyOrigin],
    block_scales: UnsafePointer[Float32, MutAnyOrigin],
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    block_colsums: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    debug_assert(K % VNNI_K_STEP == 0, "K must be a multiple of VNNI_K_STEP")
    debug_assert(K % fwht_blk == 0, "K must be a multiple of fwht_blk")
    debug_assert(N % VNNI_N_STEP == 0, "N must be a multiple of VNNI_N_STEP")
    debug_assert(VNNI_K_STEP % fwht_blk == 0, "VNNI_K_STEP must be a multiple of fwht_blk")

    comptime width = gemv_tile_width[DType.int32, VNNI_TILE_N]()
    comptime regs = VNNI_TILE_N // width
    comptime dc_per_kstep = VNNI_K_STEP // VNNI_BLK
    comptime dc_per_fwht_blk = fwht_blk // VNNI_BLK
    comptime sub_blks_per_kstep = VNNI_K_STEP // fwht_blk

    var n_block = compute_n_block(N, K)
    var packed_off = 0

    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var f32_t0 = SIMD[DType.float32, VNNI_TILE_N](0)
            var f32_t1 = SIMD[DType.float32, VNNI_TILE_N](0)

            for ks in range(0, K, VNNI_K_STEP):
                var t0 = InlineArray[SIMD[DType.int32, VNNI_TILE_N], sub_blks_per_kstep](
                    fill=SIMD[DType.int32, VNNI_TILE_N](0))
                var t1 = InlineArray[SIMD[DType.int32, VNNI_TILE_N], sub_blks_per_kstep](
                    fill=SIMD[DType.int32, VNNI_TILE_N](0))

                # tile0 — all dc iterations contiguous in 6D packed layout
                for dc in range(dc_per_kstep):
                    var sb = dc // dc_per_fwht_blk
                    var k_pos = ks + dc * VNNI_BLK
                    var acc = t0[sb]
                    comptime for r in range(regs):
                        acc = acc.insert[offset=r * width](
                            dot[width](acc.slice[width, offset=r * width](),
                                act_row, wpacked + packed_off + r * width * VNNI_BLK, k_pos))
                    t0[sb] = acc
                    packed_off += VNNI_TILE_N * VNNI_BLK

                # tile1 — all dc iterations contiguous in 6D packed layout
                for dc in range(dc_per_kstep):
                    var sb = dc // dc_per_fwht_blk
                    var k_pos = ks + dc * VNNI_BLK
                    var acc = t1[sb]
                    comptime for r in range(regs):
                        acc = acc.insert[offset=r * width](
                            dot[width](acc.slice[width, offset=r * width](),
                                act_row, wpacked + packed_off + r * width * VNNI_BLK, k_pos))
                    t1[sb] = acc
                    packed_off += VNNI_TILE_N * VNNI_BLK

                # Dequantize per fwht_blk
                for sb in range(sub_blks_per_kstep):
                    var blk_idx = ks // fwht_blk + sb
                    var dq = block_scales[blk_idx] / 127.0
                    var n0 = nb + ns
                    var cs0 = (block_colsums + blk_idx * N + n0).load[width=VNNI_TILE_N]()
                    var cs1 = (block_colsums + blk_idx * N + n0 + VNNI_TILE_N).load[width=VNNI_TILE_N]()
                    f32_t0 += (t0[sb].cast[DType.float32]() - 128.0 * cs0) * dq
                    f32_t1 += (t1[sb].cast[DType.float32]() - 128.0 * cs1) * dq

            var n0 = nb + ns
            (dst + n0).store(f32_t0 * (wsc + n0).load[width=VNNI_TILE_N]())
            (dst + n0 + VNNI_TILE_N).store(f32_t1 * (wsc + n0 + VNNI_TILE_N).load[width=VNNI_TILE_N]())


# --- helpers ---


def max_abs_diff(a: UnsafePointer[Float32, MutAnyOrigin],
                 b: UnsafePointer[Float32, MutAnyOrigin], n: Int) -> Float32:
    var mx = Float32(0)
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    return mx


def run_test[N: Int, K: Int, BLK: Int](label: String):
    print("--- " + label + " ---")
    var td = TestData[N, K]()
    var blk_sc = make_blk_scales(K // BLK)
    var colsums = compute_colsums[N, K](td.weight_ptr(), BLK)
    var bsc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(blk_sc.unsafe_ptr()))
    var cs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(colsums.unsafe_ptr()))

    # Scalar reference
    var out_ref = List[Float32](capacity=N)
    for _ in range(N):
        out_ref.append(0)
    var ref_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(out_ref.unsafe_ptr()))
    scalar_gemv_blocked[N, K](td.act_ptr(), td.weight_ptr(), bsc, td.wscale_ptr(), ref_ptr, BLK)

    # New kernel
    var out_new = List[Float32](capacity=N)
    for _ in range(N):
        out_new.append(0)
    var new_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(out_new.unsafe_ptr()))
    gemv_row_blocked_new[N, K, BLK](td.act_ptr(), td.packed_ptr(), bsc, td.wscale_ptr(), cs, new_ptr)

    var diff = max_abs_diff(ref_ptr, new_ptr, N)
    if diff < 1.0:
        print("PASS max diff:", diff)
    else:
        print("FAIL max diff:", diff)
        for i in range(min(4, N)):
            print("  [", i, "] ref:", ref_ptr[i], " new:", new_ptr[i])


def run_test_vs_standard[N: Int, K: Int, BLK: Int](label: String):
    print("--- " + label + " ---")
    var td = TestData[N, K]()
    var blk_sc = make_blk_scales(K // BLK)
    var colsums = compute_colsums[N, K](td.weight_ptr(), BLK)
    var bsc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(blk_sc.unsafe_ptr()))
    var cs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(colsums.unsafe_ptr()))

    # Standard kernel
    var out_std = List[Float32](capacity=N)
    for _ in range(N):
        out_std.append(0)
    var std_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(out_std.unsafe_ptr()))
    gemv_row_blocked[N, K, BLK](td.act_ptr(), td.packed_ptr(), bsc, td.wscale_ptr(), cs, std_ptr)

    # New kernel
    var out_new = List[Float32](capacity=N)
    for _ in range(N):
        out_new.append(0)
    var new_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(out_new.unsafe_ptr()))
    gemv_row_blocked_new[N, K, BLK](td.act_ptr(), td.packed_ptr(), bsc, td.wscale_ptr(), cs, new_ptr)

    var diff = max_abs_diff(std_ptr, new_ptr, N)
    if diff == 0:
        print("PASS (exact match)")
    else:
        print("FAIL max diff:", diff)
        for i in range(min(4, N)):
            print("  [", i, "] std:", std_ptr[i], " new:", new_ptr[i])


def run_test_deployed[N: Int, K: Int, BLK: Int](label: String):
    """Deployed gemv_row_blocked_wa must exactly match local gemv_row_blocked_new."""
    print("--- " + label + " ---")
    var td = TestData[N, K]()
    var blk_sc = make_blk_scales(K // BLK)
    var colsums = compute_colsums[N, K](td.weight_ptr(), BLK)
    var bsc = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(blk_sc.unsafe_ptr()))
    var cs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(colsums.unsafe_ptr()))

    var out_local = List[Float32](capacity=N)
    var out_deployed = List[Float32](capacity=N)
    for _ in range(N):
        out_local.append(0)
        out_deployed.append(0)
    var local_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(out_local.unsafe_ptr()))
    var deployed_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(out_deployed.unsafe_ptr()))

    gemv_row_blocked_new[N, K, BLK](td.act_ptr(), td.packed_ptr(), bsc, td.wscale_ptr(), cs, local_ptr)
    gemv_row_blocked_wa[N, K, BLK](td.act_ptr(), td.packed_ptr(), bsc, td.wscale_ptr(), cs, deployed_ptr)

    var diff = max_abs_diff(local_ptr, deployed_ptr, N)
    if diff == 0:
        print("PASS (exact match)")
    else:
        print("FAIL max diff:", diff)
        for i in range(min(4, N)):
            print("  [", i, "] local:", local_ptr[i], " deployed:", deployed_ptr[i])


def main():
    run_test_vs_standard[64, 128, 64]("blk64 new vs standard (exact)")
    run_test[64, 128, 16]("blk16 new vs scalar")
    run_test[64, 128, 32]("blk32 new vs scalar")
    run_test[64, 256, 16]("blk16 K=256 new vs scalar")
    run_test[2816, 1088, 16]("blk16 dense-size new vs scalar")
    print()
    run_test_deployed[64, 128, 64]("blk64 deployed vs local")
    run_test_deployed[64, 128, 16]("blk16 deployed vs local")
    run_test_deployed[64, 128, 32]("blk32 deployed vs local")
    run_test_deployed[64, 256, 16]("blk16 K=256 deployed vs local")
    run_test_deployed[2816, 1088, 16]("blk16 dense-size deployed vs local")
    print()
    run_e2e_test[128, 128, 128, 64, 16]("e2e blk16 no padding")
    run_e2e_test[128, 96, 128, 64, 16]("e2e blk16 padded local shard")
    run_e2e_test[1088, 1056, 256, 2816, 16]("e2e blk16 dense-size padded shard")
