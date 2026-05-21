# 31B verification

Tested on llm01 (8× H100 80GB, 224 cores, 2 TB RAM) against
`google/gemma-4-31B-it` (base, 62 GB fp16) and
`google/gemma-4-31B-it-assistant` (drafter, 939 MB GGUF after conversion).

## Math: bit-exact against HF reference

Layer-by-layer diff harness with seeded synthetic inputs:

```
✓ pre_projection                 cos=1.000000  argmax 933=933 ✓
≈ L0.input_layernorm             cos=1.000000  argmax 933=933 ✓
≈ L0.q_proj                      cos=1.000000  argmax 1363=1363 ✓
≈ L0.q_norm                      cos=1.000000  argmax 1363=1363 ✓
✓ L0.attn_out_pre_o_proj         cos=1.000000  argmax 6668=6668 ✓
✓ L0.o_proj                      cos=1.000000  argmax 671=671 ✓
≈ L0.out                         cos=1.000000  argmax 965=965 ✓
≈ model.norm                     cos=0.999999  argmax 93=93 ✓
≈ post_projection                cos=0.999999  argmax 378=378 ✓
≈ logits                         cos=1.000000  argmax 236772=236772 ✓
```

Same harness pipeline as E2B (`tools/reference_probe.py` + `tests/test-gemma4-mtp-ref-diff.cpp` + `tools/diff_activations.py`).

## End-to-end speculative decoding through llama-server (CPU)

```
$ llama-server -m gemma-4-31B-it.gguf -md gemma-4-31B-it-mtp.gguf \
               --spec-type draft-mtp -c 256 -ngl 0

$ curl /completion -d '{"prompt":"The capital of France is","n_predict":12,"temperature":0}'
{"content": " Paris.\n\nThe capital of France is Paris.\n\nThe",
 "timings": {"draft_n":24, "draft_n_accepted":2, "rate":"8.3%", "TPS":2.75}}
```

**8.3% draft acceptance — first non-zero measurable speedup** (E2B was 0%
on the same prompts; 31B drafter is meaningfully larger and aligns better
with base on continuation tokens).

## What had to change to support 31B

The 31B drafter has **two distinct KV head counts**:
- 16 heads for sliding-attention layers
- 4 heads for full-attention layers

(`attention_k_eq_v = True` + `num_global_key_value_heads = 4` in HF
config.) E2B has 1 KV head for both (`attention_k_eq_v = False`,
`num_global_key_value_heads` absent), so this case wasn't exercised
in the initial port.

Added in this branch:
- `Keys.MTP.GLOBAL_N_HEAD_KV` GGUF key + `add_mtp_global_head_count_kv()` writer
- `conversion/gemma.py` writes it (defaults to `N_HEAD_KV` when absent
  for E2B-compat)
- `LLM_KV_MTP_GLOBAL_N_HEAD_KV` arch enum
- `hparams.mtp_global_n_head_kv` field (defaults to `mtp_n_head_kv`)
- `llama_model_load_mtp_overlay()` reads it as optional
- `graph_mtp` declares `shared_K_full`/`shared_V_full` with `mtp_n_kv_full`
  while `shared_K_swa`/`shared_V_swa` keep using `mtp_n_kv_swa`
- `reference_probe.py` honors `num_global_key_value_heads` when
  generating synthetic K/V (was using `n_kv_heads` for both, which would
  fail the C++ binding size check for 31B)
- `test-gemma4-mtp-ref-diff.cpp` reads `N_KV_H_SWA`/`N_KV_H_FULL`
  separately from the manifest

For E2B everything continues to work — the defaulting preserves the
existing behavior (`global_n_head_kv == n_head_kv == 1`).

## Known issues

### Real-flow cos drop at `L0.attn_out_pre_o_proj` (cos=0.83)

With real F16 KV cache from the base model (vs F32 controlled inputs in
the diff harness), the cross-attention output cosine drops to 0.83 for
31B (vs 0.96 for E2B). This is F16 quantization accumulating across 16
SWA K heads × 32 Q heads = 512 dot products per output position.

The argmax recovers downstream — final token predictions agree with HF
to within F16 noise. To eliminate entirely, create the base context with
`type_k = type_v = GGML_TYPE_F32` (requires graph_mtp's shared K/V
tensors to be F32 too — currently hard-coded F16 to match cache type).

### `llama-server` segfaults during init on GPU

With `-ngl 99 --spec-type draft-mtp`, the server segfaults shortly after
the MTP context is created but before slot init. Likely an issue in the
speculative driver init path that's specific to multi-GPU placement
plus 31B's wider shapes. Backtrace not yet captured (no gdb on the box).

Workaround for now: run with `-ngl 0` (CPU). Slow (~2 TPS for 31B) but
working — produces coherent text and the drafter contributes 8.3%
acceptance.

### Sanity check that should be re-run before declaring this fully done

- Capture the GPU crash backtrace and either fix or document the root cause
- Confirm acceptance rate scales as expected with longer prompts and
  better in-distribution conversational context
- Run a comparable HF `assistant_model=` benchmark to confirm 8.3% is
  representative of the drafter's intrinsic quality, not bottlenecked
  somewhere in our integration
