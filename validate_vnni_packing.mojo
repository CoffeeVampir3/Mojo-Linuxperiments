"""Validate VNNI packing on real quantized model weights.

Loads the quantized safetensors twice into plain buffers:
  - unpacked: row-major int8 (reference)
  - packed: after pack_weights applies VNNI 6D transform

For each packable weight, verifies packed[vnni_offset(n,k)] == unpacked[n*K+k].
"""

from std.memory import UnsafePointer, memcpy
from std.memory.unsafe_pointer import alloc
from std.pathlib import Path

from modeling.smollm2_int8ch_tp import (
    Int8TPModel, Int8TPLayer, SmolLM2Config, pack_weights,
)
from modeling.model_spec import (
    Encoding, Shaped, Placed, Named,
    WeightDesc, weight_desc, pack_noop,
    Quantizable,
)
from modeling.loader import load_safetensors


comptime C = SmolLM2Config
comptime M = Int8TPModel[1]

comptime QUANTIZED_PATH = "quantized_scratch/model_int8_packed.safetensors"

# VNNI constants (must match kernels/vnni.mojo)
comptime TILE_N = 16
comptime VNNI_BLK = 4
comptime N_STEP = 32
comptime K_STEP = 64
comptime L2_TARGET = 256 * 1024


def compute_n_block(n: Int, k: Int) -> Int:
    var max_n = L2_TARGET // k
    var n_block = (max_n // N_STEP) * N_STEP
    if n_block >= n:
        return n
    if n_block >= N_STEP:
        return n_block
    return N_STEP


def vnni_offset(
    n: Int, k: Int,
    N: Int, K: Int,
    n_block: Int, k_block: Int,
) -> Int:
    """Scalar 6D offset matching the BufferB tile layout."""
    var n_block_idx = n // n_block
    var n_block_begin = n_block_idx * n_block
    var n_block_size = min(n_block, N - n_block_begin)

    var k_block_idx = k // k_block
    var k_block_begin = k_block_idx * k_block
    var k_block_size = min(k_block, K - k_block_begin)

    var n_within_block = n - n_block_begin
    var k_within_block = k - k_block_begin
    var n_step_idx = n_within_block // N_STEP
    var k_step_idx = k_within_block // K_STEP

    var tile_base = (
        n_block_begin * K
        + k_block_begin * n_block_size
        + n_step_idx * k_block_size * N_STEP
        + k_step_idx * K_STEP * N_STEP
    )

    var n_within_step = n_within_block % N_STEP
    var k_within_step = k_within_block % K_STEP
    var n_sub = n_within_step // TILE_N
    var n_local = n_within_step % TILE_N
    var k_vnni_local = k_within_step // VNNI_BLK
    var k_in_vnni = k_within_step % VNNI_BLK

    var sub_offset = (
        n_sub * TILE_N * K_STEP
        + k_vnni_local * TILE_N * VNNI_BLK
        + n_local * VNNI_BLK
        + k_in_vnni
    )

    return tile_base + sub_offset


def verify_weight(
    name: String,
    unpacked: UnsafePointer[UInt8, MutAnyOrigin],
    packed: UnsafePointer[UInt8, MutAnyOrigin],
    offset: Int,
    rows: Int, cols: Int,
) -> Bool:
    """Verify packed weight matches the scalar 6D formula applied to unpacked."""
    var n_block = compute_n_block(rows, cols)
    var k_block = cols
    var src = unpacked + offset
    var dst = packed + offset
    var errors = 0

    for n in range(rows):
        for k in range(cols):
            var expected = src[n * cols + k]
            var packed_off = vnni_offset(n, k, rows, cols, n_block, k_block)
            var got = dst[packed_off]
            if Int(expected) != Int(got):
                if errors < 3:
                    print(
                        "  " + name + " MISMATCH (" + String(n) + "," + String(k) + ")"
                        + " expected=" + String(Int(expected))
                        + " got=" + String(Int(got))
                        + " offset=" + String(packed_off)
                    )
                errors += 1

    if errors == 0:
        print("  " + name + " [" + String(rows) + "x" + String(cols) + "] PASS")
    else:
        print("  " + name + " FAIL: " + String(errors) + " mismatches")
    return errors == 0


def main():
    print("=== Validate VNNI packing on real model weights ===\n")

    var path = Path(QUANTIZED_PATH)

    # Allocate two buffers: one stays row-major, one gets packed
    var arena_size = M.host_arena_bytes()
    var unpacked = alloc[UInt8](arena_size)
    var packed = alloc[UInt8](arena_size)

    # Load into both
    var bases_unpacked = List[Int]()
    bases_unpacked.append(Int(unpacked))
    var bases_packed = List[Int]()
    bases_packed.append(Int(packed))

    var r1 = load_safetensors[M](path, bases_unpacked)
    if not r1:
        print("failed to load unpacked")
        return
    var r2 = load_safetensors[M](path, bases_packed)
    if not r2:
        print("failed to load packed")
        return

    print("loaded " + String(r1.value().bytes_loaded) + " bytes\n")

    # Pack the second copy using the same pack_weights the model uses.
    # Use scratch from beyond the weight region (same as SmolLM2Int8.load does).
    var scratch = UnsafePointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(packed) + M.DISTRIBUTED_BYTES + M.SCRATCH_OFF
    )
    pack_weights[M](Int(packed), scratch)

    # Verify each packable weight
    var all_pass = True

    @parameter
    def check[T: Encoding & Shaped & Placed & Named](prefix: String, base: Int):
        comptime if conforms_to(T, Quantizable):
            var ok = verify_weight(
                prefix + String(T.NAME),
                unpacked, packed,
                base + T.OFFSET,
                T.ROWS, T.COLS,
            )
            if not ok:
                all_pass = False

    M.for_each_weight[check]()

    print()
    if all_pass:
        print("ALL WEIGHTS PASS")
    else:
        print("SOME WEIGHTS FAILED")

    unpacked.free()
    packed.free()
