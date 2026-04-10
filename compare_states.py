"""Compare bf16 vs quantized model states dumped as binary.

Each file contains: embed + 30 layers × 2 checkpoints (post-attn, post-ffn).
That's 61 snapshots of bf16[2816] = 61 × 2816 × 2 bytes = 343,552 bytes.
"""
import numpy as np
import sys

HIDDEN = 2816
NUM_LAYERS = 30

def load_states(path):
    raw = np.fromfile(path, dtype=np.float16)  # bf16 reads as float16 in numpy
    # Actually bf16 != float16. We need to handle bf16 properly.
    raw_bytes = np.fromfile(path, dtype=np.uint16)
    # Convert bf16 to float32: shift left by 16 bits
    f32 = np.zeros(len(raw_bytes), dtype=np.float32)
    f32.view(np.uint32)[:] = raw_bytes.astype(np.uint32) << 16
    return f32.reshape(-1, HIDDEN)

def cosine_sim(a, b):
    dot = np.sum(a * b)
    na = np.sqrt(np.sum(a * a))
    nb = np.sqrt(np.sum(b * b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return dot / (na * nb)

def main():
    bf16_path = "/home/grail/Desktop/Mojo-Linuxperiments/bf16_states.bin"
    quant_path = "/home/grail/Desktop/Mojo-Linuxperiments/quant_states.bin"

    bf16 = load_states(bf16_path)
    quant = load_states(quant_path)

    print(f"bf16:  {bf16.shape} ({bf16.nbytes} bytes from file)")
    print(f"quant: {quant.shape} ({quant.nbytes} bytes from file)")

    n = min(bf16.shape[0], quant.shape[0])

    # Row 0 = embed, then pairs of (post-attn, post-ffn) per layer
    labels = ["embed"]
    for i in range(NUM_LAYERS):
        is_full = (i + 1) % 6 == 0
        ltype = "full" if is_full else "slid"
        labels.append(f"L{i:2d} {ltype} attn")
        labels.append(f"L{i:2d} {ltype} ffn ")

    print(f"\n{'checkpoint':<20s} {'cosine':>10s} {'max_err':>10s} {'rms_err':>10s} {'bf16_norm':>10s} {'qnt_norm':>10s}")
    print("-" * 72)

    for i in range(min(n, len(labels))):
        a = bf16[i].astype(np.float64)
        b = quant[i].astype(np.float64)
        cos = cosine_sim(a, b)
        diff = a - b
        max_err = np.max(np.abs(diff))
        rms_err = np.sqrt(np.mean(diff * diff))
        a_norm = np.sqrt(np.sum(a * a))
        b_norm = np.sqrt(np.sum(b * b))

        label = labels[i] if i < len(labels) else f"row {i}"
        print(f"{label:<20s} {cos:10.6f} {max_err:10.3f} {rms_err:10.3f} {a_norm:10.3f} {b_norm:10.3f}")

if __name__ == "__main__":
    main()
