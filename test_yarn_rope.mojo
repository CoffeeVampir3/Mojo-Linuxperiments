"""Validate YaRN RoPE against Python reference data."""

from std.pathlib import Path
from std.memory import UnsafePointer, alloc
from std.sys.info import simd_width_of

from modeling.model_spec import Encoding, Shaped, Bound, DynView, Slot, Replicated, BF16, F32
from kernels.kv_rotors import init_rope_tables, rope_apply, yarn_mscale, yarn_softmax_scale

# V2-Lite YaRN config
comptime DIM = 64
comptime HALF = DIM // 2
comptime Q_HEAD_DIM = 192
comptime MAX_POS = 8192

comptime NUM_TEST_POS = 6
comptime REF_DIR = "validation/reference_data"

comptime CosSlot = Slot[F32, Replicated, MAX_POS, HALF, 1]
comptime SinSlot = Slot[F32, Replicated, MAX_POS, HALF, 1]


def read_file_bytes(path: Path) -> List[UInt8]:
    var result = List[UInt8]()
    try:
        var data = path.read_bytes()
        for i in range(len(data)):
            result.append(data[i])
    except e:
        print("Failed to read", path, ":", e)
    return result^


def main():
    print("=== YaRN RoPE Mojo Validation ===")
    print()

    # Step 1: mscale validation
    var ms = yarn_mscale(40.0, 0.707)
    var ss = yarn_softmax_scale(Q_HEAD_DIM, 40.0, 0.707)
    print("yarn_mscale:", ms)
    print("  expected:  1.2608037774")
    print("softmax_scale:", ss)
    print("  expected:    0.1147213868")
    print()

    # Step 2: generate tables
    var cos_ptr = alloc[Float32](MAX_POS * HALF)
    var sin_ptr = alloc[Float32](MAX_POS * HALF)

    init_rope_tables(
        Bound[CosSlot](Int(cos_ptr)),
        Bound[SinSlot](Int(sin_ptr)),
        theta=10000.0,
        factor=40.0,
        original_max_pos=4096,
        beta_fast=32.0,
        beta_slow=1.0,
    )

    # Step 3: load reference data
    var cos_bytes = read_file_bytes(Path(REF_DIR) / "yarn_cos.bin")
    var sin_bytes = read_file_bytes(Path(REF_DIR) / "yarn_sin.bin")
    var pos_bytes = read_file_bytes(Path(REF_DIR) / "yarn_test_positions.bin")

    if len(cos_bytes) == 0 or len(sin_bytes) == 0 or len(pos_bytes) == 0:
        print("Failed to load reference data.")
        print("Run: cd validation && uv run python yarn_rope_check.py")
        cos_ptr.free()
        sin_ptr.free()
        return

    var ref_cos = UnsafePointer(to=cos_bytes[0]).bitcast[Float32]()
    var ref_sin = UnsafePointer(to=sin_bytes[0]).bitcast[Float32]()
    var test_pos = UnsafePointer(to=pos_bytes[0]).bitcast[Int32]()

    # Step 4: compare
    print("=== cos/sin table comparison ===")
    var max_cos_err = Float32(0)
    var max_sin_err = Float32(0)
    var all_pass = True

    for pi in range(NUM_TEST_POS):
        var pos = Int(test_pos[pi])
        var cos_err = Float32(0)
        var sin_err = Float32(0)

        for j in range(HALF):
            var our_c = cos_ptr[pos * HALF + j]
            var ref_c = ref_cos[pi * HALF + j]
            var our_s = sin_ptr[pos * HALF + j]
            var ref_s = ref_sin[pi * HALF + j]

            var ce = abs(our_c - ref_c)
            var se = abs(our_s - ref_s)
            if ce > cos_err:
                cos_err = ce
            if se > sin_err:
                sin_err = se

        if cos_err > max_cos_err:
            max_cos_err = cos_err
        if sin_err > max_sin_err:
            max_sin_err = sin_err

        var ok = cos_err < 2e-6 and sin_err < 2e-6
        if not ok:
            all_pass = False
        print(
            "  pos", pos,
            "PASS" if ok else "FAIL",
            "| cos_err:", cos_err,
            "| sin_err:", sin_err,
        )

        if pi < 3:
            print("    ours cos[:4]:", cos_ptr[pos * HALF], cos_ptr[pos * HALF + 1], cos_ptr[pos * HALF + 2], cos_ptr[pos * HALF + 3])
            print("    ref  cos[:4]:", ref_cos[pi * HALF], ref_cos[pi * HALF + 1], ref_cos[pi * HALF + 2], ref_cos[pi * HALF + 3])

    print()
    print("max cos error:", max_cos_err)
    print("max sin error:", max_sin_err)

    if all_pass:
        print("ALL POSITIONS PASS")
    else:
        print("SOME POSITIONS FAILED")

    # Step 5: end-to-end rotation test (bf16 in → f32 math → bf16 out)
    print()
    print("=== rotation (bf16 end-to-end) ===")

    var input_bytes = read_file_bytes(Path(REF_DIR) / "yarn_rope_input.bin")
    if len(input_bytes) == 0:
        print("No rotation input data.")
    else:
        var rot_all_pass = True
        var rot_pos_list = List[Int]()
        rot_pos_list.append(0)
        rot_pos_list.append(1)
        rot_pos_list.append(42)

        for ri in range(3):
            var pos = rot_pos_list[ri]

            # Copy bf16 input into working buffer (rope_apply is in-place)
            var work = alloc[Scalar[DType.bfloat16]](DIM)
            var src = UnsafePointer(to=input_bytes[0]).bitcast[Scalar[DType.bfloat16]]()
            for i in range(DIM):
                work[i] = src[i]

            # Apply rotation: 1 block of DIM elements, stride=DIM, offset=0
            rope_apply[DIM, 1, DIM, 0](
                work, DIM, 1,
                Bound[CosSlot](Int(cos_ptr)),
                Bound[SinSlot](Int(sin_ptr)),
                pos,
            )

            # Load reference output
            var ref_name = "yarn_rope_out_pos" + String(pos) + ".bin"
            var ref_bytes = read_file_bytes(Path(REF_DIR) / ref_name)
            if len(ref_bytes) == 0:
                print("  pos", pos, "- no reference data")
                work.free()
                continue

            var ref_out = UnsafePointer(to=ref_bytes[0]).bitcast[Scalar[DType.bfloat16]]()

            # Compare bf16 outputs
            var max_err = Float32(0)
            var mismatches = 0
            for i in range(DIM):
                var ours = Float32(work[i])
                var ref_v = Float32(ref_out[i])
                var err = abs(ours - ref_v)
                if err > max_err:
                    max_err = err
                if work[i] != ref_out[i]:
                    mismatches += 1

            var bf16_ulp = Float32(0.0078125)
            var ok = max_err <= bf16_ulp
            if not ok:
                rot_all_pass = False
            print(
                "  pos", pos,
                "PASS" if ok else "FAIL",
                "| max_err:", max_err,
                "| bf16_ulp:", bf16_ulp,
                "| exact_mismatches:", mismatches, "of", DIM,
            )

            if ri == 1:
                print("    ours[:4]:", Float32(work[0]), Float32(work[1]), Float32(work[2]), Float32(work[3]))
                print("    ref [:4]:", Float32(ref_out[0]), Float32(ref_out[1]), Float32(ref_out[2]), Float32(ref_out[3]))

            work.free()
            _ = ref_bytes

        if rot_all_pass:
            print("ALL ROTATIONS PASS")
        else:
            print("SOME ROTATIONS FAILED")

    cos_ptr.free()
    sin_ptr.free()
    _ = cos_bytes
    _ = sin_bytes
    _ = pos_bytes
    _ = input_bytes
