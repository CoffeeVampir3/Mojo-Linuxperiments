# experimental5: Offline Quantization + Weight Packing Pipeline

## Intent

Build a streaming quantizer that reads a bf16 safetensors model and writes a new safetensors file containing int8 channelwise-quantized weights with offline-packed memory layouts optimized for the target GEMV micro-kernel. The output file is directly loadable by the existing inference engine with no runtime transformation — weights are loaded via mmap and used as-is.

The design principle: **build a composable pipeline of offline transformations that can be applied selectively**. Quantization, scale computation, memory layout reordering, norm absorption, and future transforms (SmoothQuant calibration, VNNI packing, blockwise grouping) are all independent stages that compose freely. Which stages to apply is a configuration decision — the infrastructure should make it easy to add, remove, or reorder them without touching the rest of the system.

---

## Architecture

### Streaming Pipeline

The quantizer processes one tensor at a time. Peak memory is bounded by the largest single tensor, not the full model.

```
Source safetensors (bf16)
    │
    ├─ Read header → compute complete output layout (deterministic from metadata)
    │
    ├─ Write output header (all offsets known before touching weight data)
    │
    └─ For each tensor:
         │
         ├─ Read bf16 data from source at known offset
         │
         ├─ Pipeline: [stage1] → [stage2] → ... → [stageN]
         │    │
         │    ├─ Norm absorption (optional): W*[k,n] = g[k] * W[k,n]
         │    ├─ Scale computation: scale[n] = max(|W[:,n]|) / 127
         │    ├─ Quantization: W_q[k,n] = clamp(round(W[k,n] / scale[n]), -128, 127)
         │    └─ Layout reorder: pack W_q into micro-kernel-optimal memory order
         │
         ├─ Write packed int8 weights to output
         └─ Write f32 scales to output
```

### What Gets Quantized

| Tensor class | Treatment |
|---|---|
| Projection weights (q/k/v/o/gate/up/down_proj) | int8 channelwise + layout reorder |
| Embedding table (embed_tokens.weight) | Passthrough as bf16 (or optionally quantize) |
| Norm weights (input_layernorm, post_attention_layernorm, final norm) | Passthrough as bf16 — or absorbed into projections if norm absorption is enabled |
| RoPE tables | Not in the safetensors file (precomputed at init) |

### Output File Format

Standard safetensors. The header maps tensor names to dtype/shape/offset. Scale tensors are stored alongside their weights with a `_scale` suffix:

```json
{
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "I8", "shape": [576, 576], "data_offsets": [0, 331776]
  },
  "model.layers.0.self_attn.q_proj.weight_scale": {
    "dtype": "F32", "shape": [576, 1], "data_offsets": [331776, 334080]
  },
  ...
}
```

---

## Offline Pipeline Stages

### Stage 1: Norm Absorption (Optional)

Multiply each projection weight column by its preceding RMSNorm gain, eliminating the normalization as a runtime operation.

```
W*[k,n] = g[k] * W[k,n]
```

Performed at f32 precision to avoid bf16 rounding before quantization. The norm weight tensors are omitted from the output file (or zeroed) since their information is absorbed.

At inference, the RMSNorm call is replaced by a scalar RMS reduction on the input vector and a single multiply on the GEMV output. The gain application and intermediate normalized buffer are eliminated entirely.

**Reference**: FlashNorm (arXiv:2407.09577) — proves algebraic equivalence for bias-free linear layers. LLM Inference Acceleration via Efficient Operation Fusion (arXiv:2502.17728) — full system integration showing 15-20% latency reduction.

**Consideration**: The absorption changes the weight magnitude distribution per channel. For int8 quantization, this is benign — the per-channel scale naturally adapts to the absorbed range. In fact, the absorbed weight may have a *tighter* distribution if the gain normalizes outlier channels, potentially improving quantization quality.

### Stage 2: Channelwise Int8 Quantization

Per output channel (per row, since weights are stored [rows, cols]):

```
scale[n] = max(|W[n, :]|) / 127.0
W_q[n, k] = clamp(round(W[n, k] / scale[n]), -128, 127)
```

The scale is a single f32 value per output channel. For SmolLM2-135M, the largest projection is 1536x576 (gate/up_proj), giving 1536 scales = 6KB. Negligible overhead.

The GEMV dequantization is a single multiply on the output vector:

```
y[n] = (Σ_k x_q[k] * W_q[n,k]) * scale_x * scale_w[n]
```

The int32 accumulation is exact. The only approximation is in the initial quantization rounding.

**Error bound**: Per-element quantization error is bounded by `scale/2`. For a dot product over K elements, errors are approximately independent, giving total error O(sqrt(K) * scale/2). With K=576 and typical scale ~0.01, the dot product error is ~0.12 — small relative to activation magnitudes of 1-10.

**Reference**: SmoothQuant (arXiv:2211.10438) — while we don't use the smoothing transformation, Section 3 provides rigorous analysis of per-channel quantization error. Also: the int8 quantization scheme is standard symmetric affine quantization as described in Dettmers et al. "GPT3.int8()" (NeurIPS 2022).

### Stage 3: Weight Layout Reordering (Offline Packing)

The default weight layout is row-major: `W[n, k]` stores output channel n's weights contiguously. The GEMV micro-kernel processes Nr output channels simultaneously, streaming K elements for each. In row-major layout, accessing Nr channels at the same K position requires Nr strided loads.

Offline packing reorders weights into **panel-major** order matching the micro-kernel's access pattern. For a micro-kernel with Nr=4 and SIMD width=16 (processing 16 int8 values per lane):

```
Standard row-major:
  W[0, 0..K], W[1, 0..K], W[2, 0..K], W[3, 0..K], ...

Panel-major (Nr=4, packed for SIMD):
  W[0,0] W[1,0] W[2,0] W[3,0]  W[0,1] W[1,1] W[2,1] W[3,1]  ...  W[0,15] W[1,15] W[2,15] W[3,15]
  W[0,16] W[1,16] W[2,16] W[3,16] ...
  (next panel: W[4,0] W[5,0] W[6,0] W[7,0] ...)
```

A single SIMD load now fetches int8 weights for Nr channels at consecutive K positions — sequential memory, no strides, no TLB misses. The scale factors can be co-located at the panel boundary so the dequantization scale is adjacent to its weights in memory.

**VNNI-optimal packing**: Intel VNNI (`vpdpbusd`) performs 4 × (4 int8 multiplies + pairwise add → int32 accumulate) in one instruction. The operand layout expects 4 consecutive int8 values from the activation and 4×4 int8 values from the weight arranged in a specific interleaving. The offline packer arranges weight bytes to match this layout exactly, so the inner loop is a tight sequence of load + vpdpbusd with no shuffling.

**Reference**:

- Goto & Van de Geijn, "Anatomy of High-Performance Matrix Multiplication" (ACM TOMS, 2008) — foundational analysis of why packing is necessary and how tile sizes relate to cache hierarchy. The analytical model for choosing mc, nc, kc from cache capacities applies directly.
- BLIS Framework (Van Zee & van de Geijn, ACM TOMS, 2015) — the packing strategy generalized across architectures. The key insight: packing cost is O(K*N) but is amortized over O(M*K*N) compute, so it's free for GEMM. For GEMV (M=1), packing must be offline since there's no compute to amortize against.
- "Highly Optimized Kernels and Fine-Grained Codebooks for LLM Inference on Arm CPUs" (arXiv:2501.00032, Dec 2024) — describes this exact approach for quantized LLM GEMV: weight interleaving matched to the SIMD compute order, with scales co-located alongside weight blocks.
- CAKE: "Matrix Multiplication Using Constant-Bandwidth Blocks" (Kung et al., SC'21) — analytical framework for choosing tile sizes that achieve constant bandwidth at every cache level, giving closed-form expressions for optimal packing dimensions.
- BLISlab (arXiv:1609.00076) — step-by-step tutorial implementing the Goto algorithm with packing, suitable for adapting to GEMV.

---

## Composability

The pipeline stages are independent transforms on a weight tensor. They compose in order:

```
bf16 source weights
    → [norm absorption]     (optional, requires norm weight)
    → [scale computation]   (reads f32 weights, produces f32 scale vector)
    → [quantization]        (reads f32 weights + scale, produces int8 weights)
    → [layout reorder]      (reads int8 weights, produces packed int8 weights)
    → output
```

Each stage is a pure function: `(input_tensor, config) → output_tensor`. No stage depends on any other tensor or any model execution. This means:

- Stages can be added/removed via configuration (e.g., skip norm absorption, skip reordering)
- Future stages slot in naturally (SmoothQuant smoothing, blockwise grouping, 4-bit packing)
- The quantizer is testable per-stage (verify each transform independently)
- The entire pipeline is streaming — one tensor at a time, bounded memory

### Future Stages (Not in Initial Implementation)

**SmoothQuant smoothing**: Requires calibration data and model execution to compute per-layer `s` vectors. Would insert before scale computation: `W'[k,n] = W[k,n] * s[k]`. The `s` vectors could be precomputed by a separate calibration tool and passed to the quantizer as an input file. This decouples calibration from quantization.

**Blockwise quantization**: Instead of one scale per channel, one scale per (channel, block) where block_size divides K. The quantization stage partitions each channel into blocks and computes a local scale. The scale tensor shape changes from [N, 1] to [N, K/block_size]. The GEMV kernel dequantizes per-block instead of per-channel.

**4-bit packing**: Two int4 values per byte. The quantization stage produces 4-bit values, then a packing stage interleaves them into bytes. Requires a different GEMV kernel for unpacking.

---

## Infrastructure Needed

### Safetensors Writer

The safetensors file format is: `[8-byte u64 LE header length][JSON header][raw data]`.

We need:
- `write_u64_le(value) → List[Byte]`
- `build_safetensors_header(entries: List[TensorWriteEntry]) → String` — deterministic JSON construction via string concatenation (no general JSON serializer needed)
- `write_safetensors_file(path, header, data_segments)` — sequential write via stdlib `open()` or linux `sys_openat` + `sys_write`

The header is constructible from tensor metadata alone (name, dtype, shape, cumulative offset). All sizes are computable before reading any weight data.

### Quantization Kernels

- `compute_channel_scales(src: UnsafePointer[bfloat16], rows: Int, cols: Int, dst_scales: UnsafePointer[Float32])` — per-row reduce-max, divide by 127
- `quantize_channelwise(src: UnsafePointer[bfloat16], scales: UnsafePointer[Float32], rows: Int, cols: Int, dst: UnsafePointer[Int8])` — scale + round + clamp
- `reorder_panel(src: UnsafePointer[Int8], rows: Int, cols: Int, nr: Int, dst: UnsafePointer[Int8])` — row-major to panel-major reorder

All are BurstPool-dispatchable (partition across rows).

### Validation

The existing inference pipeline provides end-to-end validation:
1. Run bf16 model on a prompt, capture logits
2. Run int8 model on the same prompt, capture logits
3. Compare: max absolute error, cosine similarity, top-k agreement
4. Decode both and compare generated text

The tokenizer, safetensors loader, and forward pass are all already in the repo.

---

## File Layout

```
experimental5/
    design_doc.md           — this document
    quantize.mojo           — quantizer entry point (streaming pipeline)
    quant_kernels.mojo      — quantization + reordering kernels
    safetensors_writer.mojo — minimal safetensors file writer
    int8_model_spec.mojo    — int8 model layout (extends experimental4 model_spec)
    int8_kernel_ops.mojo    — int8 GEMV kernel (VNNI-optimal inner loop)
    int8_smollm2.mojo       — int8 inference entry point (forward pass using int8 kernels)
```
