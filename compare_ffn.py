"""Compare bf16 vs quantized FFN intermediates for layer 0.

Files: {bf16|quant}_L0_ffn_{dense|moe}.bin
Each file: bf16[2816] = 5632 bytes.
"""
import numpy as np
import sys

HIDDEN = 2816

def load_bf16(path):
    raw = np.fromfile(path, dtype=np.uint16)
    f32 = np.zeros(len(raw), dtype=np.float32)
    f32.view(np.uint32)[:] = raw.astype(np.uint32) << 16
    return f32

def cosine_sim(a, b):
    dot = np.sum(a * b)
    na = np.sqrt(np.sum(a * a))
    nb = np.sqrt(np.sum(b * b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return dot / (na * nb)

def compare(label, bf16_path, quant_path):
    try:
        a = load_bf16(bf16_path).astype(np.float64)
        b = load_bf16(quant_path).astype(np.float64)
    except Exception as e:
        print(f"  {label}: SKIP ({e})")
        return

    if len(a) != HIDDEN or len(b) != HIDDEN:
        print(f"  {label}: wrong size bf16={len(a)} quant={len(b)} (expected {HIDDEN})")
        return

    cos = cosine_sim(a, b)
    diff = a - b
    max_err = np.max(np.abs(diff))
    rms_err = np.sqrt(np.mean(diff * diff))
    a_norm = np.sqrt(np.sum(a * a))
    b_norm = np.sqrt(np.sum(b * b))

    print(f"  {label}:")
    print(f"    cosine:   {cos:.6f}")
    print(f"    max_err:  {max_err:.4f}")
    print(f"    rms_err:  {rms_err:.4f}")
    print(f"    bf16_norm: {a_norm:.4f}  quant_norm: {b_norm:.4f}")
    print(f"    bf16[0:4]: {a[:4]}")
    print(f"    qnt[0:4]:  {b[:4]}")

    # Distribution analysis
    print(f"    bf16 range: [{np.min(a):.4f}, {np.max(a):.4f}]  mean: {np.mean(a):.4f}")
    print(f"    qnt range:  [{np.min(b):.4f}, {np.max(b):.4f}]  mean: {np.mean(b):.4f}")

def main():
    base = "/home/grail/Desktop/Mojo-Linuxperiments"
    print("=== Layer 0 FFN Intermediate Comparison ===\n")

    compare("Dense output",
        f"{base}/bf16_L0_ffn_dense.bin",
        f"{base}/quant_L0_ffn_dense.bin")

    print()
    compare("MoE accumulated",
        f"{base}/bf16_L0_ffn_moe.bin",
        f"{base}/quant_L0_ffn_moe.bin")

if __name__ == "__main__":
    main()
