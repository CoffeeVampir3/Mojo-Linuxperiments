# Future Considerations

## i8 V-agg for Single-Token Decode

### Summary

At sl=1 (single-token decode), the i8 W quantize + tdpbusd V-agg strategy
outperforms bf16 by 1.6-2x at meaningful context lengths. This is because:

- V is read directly from the u8 KV cache as the A operand (zero conversion)
- W (softmax weights) are quantized to i8 and VNNI-packed inline during softmax
- No per-chunk V conversion or VNNI repacking overhead to amortize
- The position-outer loop order avoids redundant KV processing

The bf16 path (KV-outer loop) pays per-chunk costs (K VNNI pack, V bf16
convert + VNNI pack) that only amortize when multiple query positions reuse
them. At sl=1 there is nothing to amortize.

### Benchmark Data (128h/8kv, hd=128, 4 NUMA nodes)

```
ctx  | sl | i8 us/pos | bf16 us/pos | i8 speedup
-----|----|-----------:|------------:|-----------:
 128 |  1 |       277 |         290 |      1.0x
2048 |  1 |       384 |         602 |      1.6x
4096 |  1 |       527 |         990 |      1.9x
 128 | 64 |        45 |          26 |      0.6x
2048 | 64 |       112 |          85 |      0.7x
4096 | 64 |       155 |         129 |      0.8x
```

The crossover is around sl=8-16. Below that, i8 wins. Above, bf16 wins.

### Architecture (committed version as of 2026-03-31)

The i8 decode kernel (`hadquant_attn_amx.mojo`) uses:

- **Loop order**: position-outer, group-middle, chunk-inner
- **Scoring**: `S^T[chunk, gqa] = K_u8[chunk, hd] x Q_vnni[hd, gqa]` via tdpbusd
- **Softmax**: SIMD-over-heads (TILE_N=16 = gqa_factor), fused exp + V-scale
  absorption + i8 quantize + VNNI interleave in one pass
- **V agg**: `O^T[hd, gqa] = V_u8[hd, chunk] x W_vnni[chunk, gqa]` via tdpbusd
- **Key optimization**: W quantize uses a fixed scale from v_scale_max per chunk,
  eliminating the absmax pass. Bias correction is `128 * sum(W_i8_column)`.

### When to Revisit

Consider reintroducing the i8 path if:

- Context splitting (FlashDecoding) is implemented and sl=1 decode needs
  maximum per-worker throughput
- The i8 quantize + VNNI pack cost can be reduced (e.g., via AMX-native
  int8 conversion instructions in future ISA extensions)
- A hybrid kernel can select i8 vs bf16 at runtime based on seq_len
