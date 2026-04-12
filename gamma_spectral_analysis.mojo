"""Spectral analysis of gamma within FWHT blocks.

For each norm weight, examines the per-block gamma distribution to determine
whether the conjugated operator H·diag(sqrt_gamma)·H^T has low-rank structure.

If gamma within a block is mostly constant c with k outliers, the conjugated
operator is rank-k perturbation of scalar c·I, applicable in O(n·k) instead
of O(n²) or O(n·log(n)).

Run: pixi run mojo -I . gamma_spectral_analysis.mojo
"""

from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.math import abs
from std.pathlib import Path
from std.collections import InlineArray

from simd_math import sqrt as simd_sqrt
from safetensors.parser import parse_safetensors_header, SafetensorsHeader, TensorMeta
from notstdcollections import HeapMoveArray
from modeling.loader import discover_shards, find_tensor

from experimental3.common_math import F32Ptr, BF16Ptr


comptime K = 2816
comptime BLK = 256
comptime NUM_BLK = K // BLK
comptime BF16_MODEL_DIR = "checkpoints/gemma-4-26B-A4B"


def load_bf16_tensor(
    name: String,
    ref headers: HeapMoveArray[SafetensorsHeader],
    ref shards: List[Path],
) -> Optional[BF16Ptr]:
    var found = find_tensor(name, headers)
    if not found:
        print("  missing: " + name)
        return None
    var shard_idx = found.value()[0]
    var meta = found.value()[1].copy()
    var byte_off = headers[shard_idx].data_offset + meta.start
    var nbytes = meta.end - meta.start
    try:
        with open(shards[shard_idx], "r") as f:
            _ = f.seek(UInt64(byte_off), 0)
            var data = f.read_bytes(size=nbytes)
            if len(data) != nbytes:
                return None
            var buf = alloc[Scalar[DType.bfloat16]](nbytes // 2)
            memcpy(dest=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(buf)),
                   src=UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(data.unsafe_ptr())),
                   count=nbytes)
            return BF16Ptr(unsafe_from_address=Int(buf))
    except:
        return None


def analyze_block_spectrum(gamma_f32: F32Ptr, n: Int, blk: Int):
    """For each FWHT block, compute how many eigenvalues (sqrt_gamma values)
    are needed to capture 95% and 99% of the variance from the block mean."""
    var num_blocks = n // blk
    print("  block | mean_sg  | max_sg   | cv%      | rank@95% | rank@99% | rank@99.9%")
    print("  ------|----------|----------|----------|----------|----------|----------")

    var total_rank_95 = 0
    var total_rank_99 = 0
    var total_rank_999 = 0

    for b in range(num_blocks):
        var off = b * blk
        # Compute sqrt(|gamma|) for this block
        var sg_buf = alloc[Float32](blk)
        var sgp = F32Ptr(unsafe_from_address=Int(sg_buf))
        var sg_sum = Float32(0)
        for k in range(blk):
            var v = abs(gamma_f32[off + k])
            if v < Float32(1e-10):
                v = Float32(1e-10)
            sgp[k] = simd_sqrt(v)
            sg_sum += sgp[k]
        var sg_mean = sg_sum / Float32(blk)

        # Compute deviations from mean: these are the "eigenvalue residuals"
        # The rank-k approximation keeps the k largest |dev|
        var dev_buf = alloc[Float32](blk)
        var devp = F32Ptr(unsafe_from_address=Int(dev_buf))
        var total_var = Float32(0)
        var sg_max = Float32(0)
        for k in range(blk):
            var d = sgp[k] - sg_mean
            devp[k] = d
            total_var += d * d
            if sgp[k] > sg_max:
                sg_max = sgp[k]

        var cv = Float32(0)
        if sg_mean > Float32(1e-10):
            cv = simd_sqrt(total_var / Float32(blk)) / sg_mean * Float32(100)

        # Sort deviations by magnitude (descending) — bubble sort, n=256 is fine
        var mag_buf = alloc[Float32](blk)
        var magp = F32Ptr(unsafe_from_address=Int(mag_buf))
        for k in range(blk):
            magp[k] = devp[k] * devp[k]
        for i in range(blk):
            for j in range(i + 1, blk):
                if magp[j] > magp[i]:
                    var tmp = magp[i]
                    magp[i] = magp[j]
                    magp[j] = tmp

        # Find rank needed for 95%, 99%, 99.9% of variance
        var cumsum = Float32(0)
        var rank_95 = blk
        var rank_99 = blk
        var rank_999 = blk
        for k in range(blk):
            cumsum += magp[k]
            if cumsum >= total_var * Float32(0.95) and rank_95 == blk:
                rank_95 = k + 1
            if cumsum >= total_var * Float32(0.99) and rank_99 == blk:
                rank_99 = k + 1
            if cumsum >= total_var * Float32(0.999) and rank_999 == blk:
                rank_999 = k + 1

        total_rank_95 += rank_95
        total_rank_99 += rank_99
        total_rank_999 += rank_999

        print("  " + String(b)
            + "     | " + String(sg_mean)
            + " | " + String(sg_max)
            + " | " + String(cv)
            + " | " + String(rank_95)
            + "        | " + String(rank_99)
            + "        | " + String(rank_999))

        sg_buf.free()
        dev_buf.free()
        mag_buf.free()

    print("  ------|----------|----------|----------|----------|----------|----------")
    print("  total rank across " + String(num_blocks) + " blocks: @95%="
        + String(total_rank_95) + "  @99%=" + String(total_rank_99)
        + "  @99.9%=" + String(total_rank_999)
        + "  (out of " + String(num_blocks * blk) + ")")
    print("  mean rank/block: @95%=" + String(Float32(total_rank_95) / Float32(num_blocks))
        + "  @99%=" + String(Float32(total_rank_99) / Float32(num_blocks))
        + "  @99.9%=" + String(Float32(total_rank_999) / Float32(num_blocks)))


def main():
    print("=== Gamma Spectral Analysis within FWHT Blocks ===")
    print("K=" + String(K) + "  FWHT_BLK=" + String(BLK))
    print()

    var shards = discover_shards(Path(BF16_MODEL_DIR))
    if len(shards) == 0:
        print("no shards found")
        return
    print("found " + String(len(shards)) + " shard(s)")

    var headers = HeapMoveArray[SafetensorsHeader](len(shards))
    for i in range(len(shards)):
        var h = parse_safetensors_header(shards[i])
        if not h:
            return
        headers.push(h.take())

    var norm_names = List[String]()
    var norm_labels = List[String]()
    var sample_layers = InlineArray[Int, 3](fill=0)
    sample_layers[0] = 0
    sample_layers[1] = 15
    sample_layers[2] = 29
    for li_idx in range(3):
        var li = sample_layers[li_idx]
        var prefix = "model.language_model.layers." + String(li) + "."
        norm_names.append(prefix + "input_layernorm.weight")
        norm_labels.append("L" + String(li) + " input_norm")
    norm_names.append("model.language_model.norm.weight")
    norm_labels.append("final_norm")

    var gamma_f32 = alloc[Float32](K)
    var gfp = F32Ptr(unsafe_from_address=Int(gamma_f32))

    for ni in range(len(norm_names)):
        var raw = load_bf16_tensor(norm_names[ni], headers, shards)
        if not raw:
            continue
        var bf16p = raw.value()

        for k in range(K):
            gfp[k] = Float32(bf16p[k])

        print("━━━ " + norm_labels[ni] + " ━━━")
        analyze_block_spectrum(gfp, K, BLK)
        print()

        bf16p.free()

    gamma_f32.free()
