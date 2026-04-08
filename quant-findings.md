# Quantization Findings

This note summarizes observed results from the `smollm2_butterquant_tp` validation work.
It is limited to measured behavior from the current validator and code review.

## Scope

- Model path under investigation: `modeling/smollm2_butterquant_tp.mojo`
- Reference path: bf16 `smollm2`
- Main validation harness: `butterquant/test_kernel_validation.mojo`
- Kernel files examined:
  - `experimental2/kernels/rmsnorm_fwht_quantize.mojo`
  - `experimental2/kernels/int8_gemv.mojo`
  - `experimental2/kernels/quantize.mojo`
  - `experimental2/kernels/rope_and_kv_cache_write.mojo`

The validator was used for:

- full layer-boundary comparisons across all 30 layers
- detailed layer-0 prefill breakdown
- TP=1 and TP=3 comparisons
- kernel reconstruction checks against checkpoint weights
- activation-side grouped-scale simulations

## Main Results

### 1. `v_layer_scale` was a real TP-specific bug

`v_layer_scale` originally used one TP shard while normalizing as if it represented the full `V` matrix.
After changing it to aggregate across TP shards, TP=3 moved much closer to TP=1.

Selected TP=3 before/after results:

- layer-0 `attn_block`: rel-L2 `0.17698` -> `0.01606`
- layer-0 `mlp_block`: rel-L2 `0.11595` -> `0.01705`
- layer-29 `attn`: rel-L2 `0.06860` -> `0.04375`
- layer-29 `mlp`: rel-L2 `0.10763` -> `0.06091`

After the fix:

- TP=1 layer-29 `mlp`: rel-L2 `0.06228`
- TP=3 layer-29 `mlp`: rel-L2 `0.06091`

### 2. After the `v_layer_scale` fix, TP=1 and TP=3 track closely

Layer-0 block outputs:

- TP=1 `attn_block`: rel-L2 `0.01616`
- TP=3 `attn_block`: rel-L2 `0.01606`
- TP=1 `mlp_block`: rel-L2 `0.01748`
- TP=3 `mlp_block`: rel-L2 `0.01705`

Layer-0 hybrid attribution to final layer output:

- TP=1 `attn_only_to_layer_out`: rel-L2 `0.01213`
- TP=1 `mlp_only_to_layer_out`: rel-L2 `0.01287`
- TP=3 `attn_only_to_layer_out`: rel-L2 `0.01221`
- TP=3 `mlp_only_to_layer_out`: rel-L2 `0.01253`

### 3. No runtime mismatch was found in the gate/up projection kernel

For layer-0 MLP gate/up projections, the runtime GEMV output was compared against a reconstructed qdq matmul built from:

- dequantized checkpoint weights
- correct inverse-FWHT weight basis
- quantized/dequantized activations

Results:

- `gate_proj ... kernel_vs_both_qdq`: rel-L2 about `1.5e-7`
- `up_proj ... kernel_vs_both_qdq`: rel-L2 about `2.0e-7`

This means the runtime path in `experimental2/kernels/int8_gemv.mojo` matched the intended quantized math within numerical noise for the measured cases.

### 4. `fused_gu_silu` matched the unfused computation

The fused kernel output was compared against:

1. separate gate GEMV
2. separate up GEMV
3. explicit `SiLU(gate) * up`
4. the same final absmax qdq path

Result:

- `fused_vs_sep_absmax`: rel-L2 `0.0` for TP=1 and TP=3 rank-local checks

This means the fused implementation did not add extra discrepancy beyond the same underlying quantized math.

### 5. Within the MLP block, `up_proj` had larger local error than `gate_proj`

Layer-0 TP=1:

- `gate_proj_f32`: rel-L2 `0.01667`
- `up_proj_f32`: rel-L2 `0.02514`

Layer-0 TP=3 rank-0:

- `gate_proj_f32`: rel-L2 `0.01663`
- `up_proj_f32`: rel-L2 `0.02499`

The same ordering held across TP=3 ranks.

### 6. Activation-side projection error was larger than weight-side projection error

Layer-0 TP=1:

- `gate_proj act_only`: rel-L2 `0.01505`
- `gate_proj weight_only`: rel-L2 `0.00729`
- `up_proj act_only`: rel-L2 `0.02243`
- `up_proj weight_only`: rel-L2 `0.01169`

Layer-0 TP=3 rank-0:

- `gate_proj act_only`: rel-L2 `0.01520`
- `gate_proj weight_only`: rel-L2 `0.00712`
- `up_proj act_only`: rel-L2 `0.02213`
- `up_proj weight_only`: rel-L2 `0.01160`

These measurements indicate that the activation-side approximation was the larger local term in the measured gate/up projections.

### 7. Post-SiLU qdq was not the dominant local source

Layer-0 TP=1:

- `silu_prequant_sep`: rel-L2 `0.02018`
- `fused_gu_silu`: rel-L2 `0.02225`

Layer-0 TP=3 rank-0:

- `silu_prequant_sep`: rel-L2 `0.01935`
- `fused_gu_silu`: rel-L2 `0.02071`

The final post-SiLU qdq stage added a smaller increment on top of the pre-existing gate/up projection discrepancy.

### 8. Clipped-RMS did not improve the measured post-SiLU qdq path

A clipped-RMS sweep of the form

`scale = min(absmax, alpha * rms)`

was run on real layer-0 post-SiLU activations.

TP=1:

- absmax baseline: rel-L2 `0.00927`
- best clipped-RMS result: alpha `5.0`, rel-L2 `0.02070`

TP=3 ranks showed the same pattern: the best clipped-RMS setting collapsed back to near-absmax and did not outperform the absmax baseline.

### 9. FWHT reduced activation qdq error substantially

Direct layer-0 MLP-input activation qdq diagnostics:

- no FWHT:
  - rel-L2 `0.03764`
  - mean `absmax / rms = 16.47`
  - zero fraction `7.65%`
  - saturation fraction `0.17%`
- with FWHT:
  - rel-L2 `0.00860`
  - mean `absmax / rms = 3.77`
  - zero fraction `1.28%`
  - saturation fraction `0.19%`

TP=3 reported nearly identical values.

These measurements show lower rowwise qdq error after the FWHT-based rotation in the measured MLP-input path.

### 10. Grouped activation-scale simulation improved the measured activation-side error

Two grouped-scale simulations were run on the MLP input after FWHT:

- `group3`: one scale per `3 * FWHT_BLOCK` columns
- `group1`: one scale per `FWHT_BLOCK` columns

MLP-input qdq:

- baseline rowwise FWHT scale: rel-L2 `0.00860`
- `group3`: rel-L2 `0.00643`
- `group1`: rel-L2 `0.00505`

Relative improvement vs rowwise baseline:

- `group3`: `25.2%` lower rel-L2
- `group1`: `41.3%` lower rel-L2

Layer-0 TP=1 activation-only projection error:

- `gate_proj`:
  - baseline `0.01505`
  - `group3` `0.00650`
  - `group1` `0.00510`
- `up_proj`:
  - baseline `0.02243`
  - `group3` `0.01011`
  - `group1` `0.00792`

Relative improvement vs rowwise baseline:

- TP=1 `gate_proj`:
  - `group3`: `56.8%` lower
  - `group1`: `66.1%` lower
- TP=1 `up_proj`:
  - `group3`: `54.9%` lower
  - `group1`: `64.7%` lower

Layer-0 TP=3 rank-0 activation-only projection error:

- `gate_proj`:
  - baseline `0.01520`
  - `group3` `0.00642`
  - `group1` `0.00509`
- `up_proj`:
  - baseline `0.02213`
  - `group3` `0.01000`
  - `group1` `0.00779`

Relative improvement vs rowwise baseline:

- TP=3 rank-0 `gate_proj`:
  - `group3`: `57.8%` lower
  - `group1`: `66.5%` lower
- TP=3 rank-0 `up_proj`:
  - `group3`: `54.8%` lower
  - `group1`: `64.8%` lower

These grouped-scale results were measured in simulation only. They were not implemented in the runtime kernels.

## Notes and Caveats

- Several `kv_write.*` and intermediate attention micro-checks produced `nan` or `inf` in the validator and were not used as the basis for the conclusions above.
- The strongest conclusions in this note come from:
  - full layer-boundary comparisons
  - layer-0 block-local comparisons
  - reconstructed qdq projection checks
  - direct activation qdq diagnostics
- The grouped activation-scale results are local measurements for the MLP-input and gate/up projection path. They are not full end-to-end model results.
