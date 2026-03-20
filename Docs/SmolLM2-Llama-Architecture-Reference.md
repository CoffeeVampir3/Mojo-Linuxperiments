# SmolLM2 Checkpoint Architecture Reference (LLaMA-Style Decoder)

## Scope
This document is a build reference for implementing the architecture behind the local checkpoint at `checkpoints/SmolLM2/` from scratch.

It focuses on:
- exact high-level module structure and operation order
- how each relevant config variable maps into code behavior
- tensor shapes and equations you need for a correct implementation
- where to verify each claim in source code/docs

## What This Checkpoint Is
From local config:
- `architectures: ["LlamaForCausalLM"]` (`checkpoints/SmolLM2/config.json:3`)
- `model_type: "llama"` (`checkpoints/SmolLM2/config.json:16`)
- `is_llama_config: true` (`checkpoints/SmolLM2/config.json:13`)

So this is a **LLaMA-family decoder-only causal LM** checkpoint in HF format.

Key model dimensions (`checkpoints/SmolLM2/config.json`):
- `hidden_size = 576`
- `num_hidden_layers = 30`
- `num_attention_heads = 9`
- `num_key_value_heads = 3` (GQA)
- `intermediate_size = 1536`
- `max_position_embeddings = 8192`
- `rope_theta = 100000`
- `vocab_size = 49152`
- `tie_word_embeddings = true`

Derived:
- `head_dim = hidden_size / num_attention_heads = 64`
- KV repetition factor (GQA) = `9 / 3 = 3`

Local safetensors layout matches LLaMA naming:
- per-layer attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- per-layer MLP: `gate_proj`, `up_proj`, `down_proj`
- no MoE router/expert tensors in this checkpoint

## End-to-End Forward Graph
Given `input_ids` of shape `[B, T]`:

1. Token embeddings  
   `x = embed_tokens(input_ids)` -> `[B, T, hidden_size]`

2. Build causal mask  
   Prevent each position from attending to future positions.

3. For each decoder layer (30 times):
   1. `h = RMSNorm(x)`
   2. Self-attention block:
      - Project Q, K, V
      - Apply RoPE to Q and K
      - Apply grouped-query causal attention
      - Output projection
   3. Residual add: `x = x + attn_out`
   4. `h = RMSNorm(x)`
   5. Gated MLP (SwiGLU-style):  
      `mlp_out = down_proj( SiLU(gate_proj(h)) * up_proj(h) )`
   6. Residual add: `x = x + mlp_out`

4. Final RMSNorm

5. LM head projection to logits  
   Because `tie_word_embeddings = true`, output projection uses shared embedding weights.

## Attention Details (GQA + RoPE)
For one layer, with `Hq=9`, `Hkv=3`, `D=64`:

- `Q = Wq x` -> reshape to `[B, Hq, T, D]`
- `K = Wk x` -> reshape to `[B, Hkv, T, D]`
- `V = Wv x` -> reshape to `[B, Hkv, T, D]`

Apply RoPE on `(Q, K)` head dimensions.

Repeat KV heads to match query heads:
- `K_rep = repeat_kv(K, n_rep=3)` -> `[B, 9, T, D]`
- `V_rep = repeat_kv(V, n_rep=3)` -> `[B, 9, T, D]`

Attention:
- `A = softmax((Q @ K_rep^T) / sqrt(D) + causal_mask)`
- `O = A @ V_rep`
- merge heads and apply `o_proj`

## Config Variables You Need In Code
Use these directly in your config/dataclass and module constructors:

| Config field | Why it matters in implementation |
|---|---|
| `vocab_size` | Embedding matrix rows and LM logits size |
| `hidden_size` | Model width, residual size, projection in/out dims |
| `num_hidden_layers` | Number of decoder blocks |
| `num_attention_heads` | Query heads count |
| `num_key_value_heads` | KV heads count (GQA behavior) |
| `intermediate_size` | MLP expansion dimension |
| `hidden_act` | MLP activation (`silu` here) |
| `max_position_embeddings` | Intended context window and RoPE cache sizing |
| `rope_theta` | RoPE base frequency |
| `rope_scaling` | Optional long-context scaling strategy |
| `rms_norm_eps` | RMSNorm epsilon |
| `attention_dropout` | Attention prob dropout (0.0 here) |
| `attention_bias` | Whether attention linear layers include bias |
| `mlp_bias` | Whether MLP linear layers include bias |
| `tie_word_embeddings` | Share embedding and LM head weights |
| `use_cache` | KV-cache behavior for autoregressive decoding |
| `bos_token_id`, `eos_token_id`, `pad_token_id` | Generation/tokenization boundary behavior |
| `torch_dtype` | Default checkpoint/storage dtype expectations |

Checkpoint-specific tokenizer settings that matter operationally:
- `tokenizer_class = "GPT2Tokenizer"` (`checkpoints/SmolLM2/tokenizer_config.json:151`)
- chat template with `<|im_start|>` / `<|im_end|>` (`checkpoints/SmolLM2/tokenizer_config.json:146`)
- `model_max_length = 8192` (`checkpoints/SmolLM2/tokenizer_config.json:149`)

## Minimal Build Plan (From Scratch)
Implement modules in this order:

1. `RMSNorm`
2. `RotaryEmbedding` + `apply_rotary_pos_emb`
3. `repeat_kv` helper for GQA
4. `Attention` (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
5. `MLP` (`gate_proj`, `up_proj`, `down_proj`, `SiLU`)
6. `DecoderLayer` (pre-norm + residual around attn and MLP)
7. `Model` (embedding + N layers + final norm)
8. `CausalLMHead` (tied or untied by config)
9. Optional generation cache support

## Reference Pseudocode
```python
class DecoderLayer(nn.Module):
    def forward(self, x, cos, sin, causal_mask, cache=None):
        h = rms1(x)
        q = q_proj(h).view(B, T, Hq, D).transpose(1, 2)
        k = k_proj(h).view(B, T, Hkv, D).transpose(1, 2)
        v = v_proj(h).view(B, T, Hkv, D).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        if cache is not None:
            k, v = cache.update(k, v)

        k = repeat_kv(k, Hq // Hkv)
        v = repeat_kv(v, Hq // Hkv)
        attn = softmax((q @ k.transpose(-2, -1)) / sqrt(D) + causal_mask)
        y = (attn @ v).transpose(1, 2).reshape(B, T, Hq * D)
        x = x + o_proj(y)

        h = rms2(x)
        ff = down_proj(silu(gate_proj(h)) * up_proj(h))
        x = x + ff
        return x
```

## Validation Checklist
After implementing:

1. Confirm parameter tensor names/shapes match HF checkpoint structure.
2. Verify layer count and dims (`30`, `576`, `9/3`, `1536`).
3. Run one forward pass with fixed seed and compare logits vs HF (`max_abs_diff` threshold).
4. Validate autoregressive cache path (single-token decode parity vs full-sequence decode).

## Notes on “Is This Llama 3?”
This checkpoint uses **LLaMA-family architecture**, but it is not Meta’s Llama 3 weights.

- It runs through HF `llama` model code path due `model_type: "llama"`.
- Llama 3 docs state architecture is the same family as Llama 2, but model configs/tokenizers can differ.
- SmolLM2 config and tokenizer differ materially from Meta-Llama-3 checkpoints.

## Primary Sources
Local checkpoint files:
- `checkpoints/SmolLM2/config.json`
- `checkpoints/SmolLM2/tokenizer_config.json`
- `checkpoints/SmolLM2/model.safetensors`

Hugging Face Transformers code/docs:
- LLaMA implementation (v4.42.3):  
  https://github.com/huggingface/transformers/blob/v4.42.3/src/transformers/models/llama/modeling_llama.py
- LLaMA config class (v4.42.3):  
  https://github.com/huggingface/transformers/blob/v4.42.3/src/transformers/models/llama/configuration_llama.py
- Auto classes behavior:  
  https://huggingface.co/docs/transformers/en/model_doc/auto
- LLaMA model docs:  
  https://huggingface.co/docs/transformers/v4.42.4/model_doc/llama
- Llama 3 docs (includes architecture note):  
  https://huggingface.co/docs/transformers/en/model_doc/llama3

SmolLM2 sources:
- SmolLM2 model card:  
  https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- SmolLM2 paper:  
  https://arxiv.org/abs/2502.02737
- Local SmolLM2 README mirror:  
  `checkpoints/SmolLM2/README.md`
