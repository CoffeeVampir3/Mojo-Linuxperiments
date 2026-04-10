"""Compare intermediate (after gelu*up) between bf16 and quantized models.

bf16: real-domain bf16[2112]
quant: FWHT-domain i8[2112] + f32[33] block scales (blocks of 64)

We FWHT the bf16 intermediate to bring it to the same domain, then compare.
"""
import numpy as np

INTERMEDIATE = 2112
FWHT_BLK = 64
NUM_BLOCKS = INTERMEDIATE // FWHT_BLK
DC_SCALE = 0.5

def load_bf16(path):
    raw = np.fromfile(path, dtype=np.uint16)
    f32 = np.zeros(len(raw), dtype=np.float32)
    f32.view(np.uint32)[:] = raw.astype(np.uint32) << 16
    return f32

def fwht_block(x):
    """In-place Walsh-Hadamard transform of x (must be power of 2 length)."""
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2
    return x

def cosine_sim(a, b):
    dot = np.sum(a * b)
    na = np.sqrt(np.sum(a * a))
    nb = np.sqrt(np.sum(b * b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return dot / (na * nb)

def main():
    base = "/home/grail/Desktop/Mojo-Linuxperiments"

    # Load bf16 intermediate (real domain)
    bf16_inter = load_bf16(f"{base}/bf16_L0_ffn_intermediate.bin").astype(np.float64)
    print(f"bf16 intermediate: shape={bf16_inter.shape}, norm={np.linalg.norm(bf16_inter):.4f}")
    print(f"  bf16[0:8] = {bf16_inter[:8]}")

    # FWHT the bf16 intermediate to FWHT domain
    bf16_fwht = bf16_inter.copy()
    for b in range(NUM_BLOCKS):
        blk = bf16_fwht[b*FWHT_BLK : (b+1)*FWHT_BLK]
        fwht_block(blk)
    print(f"  bf16_fwht[0:8] = {bf16_fwht[:8]}")

    # Apply DC correction to bf16_fwht (to match quantized)
    bf16_fwht_dc = bf16_fwht.copy()
    for b in range(NUM_BLOCKS):
        bf16_fwht_dc[b * FWHT_BLK] *= DC_SCALE
    print(f"  bf16_fwht_dc[0:8] = {bf16_fwht_dc[:8]}")

    # Load quantized i8 intermediate + scales
    qi8 = np.fromfile(f"{base}/quant_L0_ffn_inter_i8.bin", dtype=np.int8).astype(np.float64)
    scales = np.fromfile(f"{base}/quant_L0_ffn_inter_scales.bin", dtype=np.float32).astype(np.float64)
    print(f"\nquant i8: shape={qi8.shape}, scales shape={scales.shape}")
    print(f"  qi8[0:8] = {qi8[:8]}")
    print(f"  scales[0:4] = {scales[:4]}")

    # Dequantize: a_fp[k] = qi8[k] * scale[k // 64] / 127
    quant_fwht_dc = np.zeros(INTERMEDIATE, dtype=np.float64)
    for b in range(NUM_BLOCKS):
        quant_fwht_dc[b*FWHT_BLK : (b+1)*FWHT_BLK] = qi8[b*FWHT_BLK : (b+1)*FWHT_BLK] * scales[b] / 127.0
    print(f"  quant_dequant[0:8] = {quant_fwht_dc[:8]}")

    # Compare in FWHT+DC domain
    cos_fwht_dc = cosine_sim(bf16_fwht_dc, quant_fwht_dc)
    diff = bf16_fwht_dc - quant_fwht_dc
    print(f"\n=== FWHT+DC domain comparison ===")
    print(f"  cosine: {cos_fwht_dc:.6f}")
    print(f"  max_err: {np.max(np.abs(diff)):.6f}")
    print(f"  rms_err: {np.sqrt(np.mean(diff*diff)):.6f}")
    print(f"  bf16_norm: {np.linalg.norm(bf16_fwht_dc):.4f}  quant_norm: {np.linalg.norm(quant_fwht_dc):.4f}")

    # Also compare in real domain: inverse FWHT the dequantized i8
    # First undo DC: divide element 0 of each block by DC_SCALE
    quant_fwht = quant_fwht_dc.copy()
    for b in range(NUM_BLOCKS):
        quant_fwht[b * FWHT_BLK] /= DC_SCALE
    # FWHT is self-inverse (up to scale 1/N per block)
    quant_real = quant_fwht.copy()
    for b in range(NUM_BLOCKS):
        blk = quant_real[b*FWHT_BLK : (b+1)*FWHT_BLK]
        fwht_block(blk)
        blk /= FWHT_BLK  # normalize: FWHT(FWHT(x))/N = x

    cos_real = cosine_sim(bf16_inter, quant_real)
    diff_real = bf16_inter - quant_real
    print(f"\n=== Real domain comparison (after inverse FWHT) ===")
    print(f"  cosine: {cos_real:.6f}")
    print(f"  max_err: {np.max(np.abs(diff_real)):.6f}")
    print(f"  rms_err: {np.sqrt(np.mean(diff_real*diff_real)):.6f}")
    print(f"  bf16_norm: {np.linalg.norm(bf16_inter):.4f}  quant_norm: {np.linalg.norm(quant_real):.4f}")
    print(f"  bf16[0:4] = {bf16_inter[:4]}")
    print(f"  qnt[0:4]  = {quant_real[:4]}")

    # Check per-block quantization quality
    print(f"\n=== Per-block analysis ===")
    for b in range(min(5, NUM_BLOCKS)):
        s, e = b*FWHT_BLK, (b+1)*FWHT_BLK
        bcos = cosine_sim(bf16_fwht_dc[s:e], quant_fwht_dc[s:e])
        brms = np.sqrt(np.mean((bf16_fwht_dc[s:e] - quant_fwht_dc[s:e])**2))
        bmax_ref = np.max(np.abs(bf16_fwht_dc[s:e]))
        bmax_qi8 = np.max(np.abs(qi8[s:e]))
        print(f"  blk {b}: cos={bcos:.4f} rms={brms:.6f} ref_max={bmax_ref:.4f} qi8_max={bmax_qi8} scale={scales[b]:.4f} utilization={bmax_qi8/127:.1%}")

if __name__ == "__main__":
    main()
