# Gemma 4 MTP (Multi-Token Prediction) for llama.cpp

Implementation of Google's Gemma 4 speculative-decoding drafter — the
`*-it-assistant` variants — for llama.cpp. The drafter is loaded as an
**MTP overlay** (a small GGUF attached in-place to the base model) and runs
through the standard `llama-server` speculative-decoding pipeline.

## Status

**Numerically verified against the HuggingFace reference** for both the
E2B and 31B variants. Diff harness with seeded synthetic inputs:

| Stage | E2B cos | 31B cos | argmax match |
|---|---|---|---|
| base `last_hidden_state` (post-norm `h_t`) | 0.999999 | 0.999999 | ✓ |
| `pre_projection` | 1.000000 | 1.000000 | ✓ |
| `L0.q_proj`, `L0.q_norm`, `L0.input_layernorm` | 1.000000 | 1.000000 | ✓ |
| `L0.attn_out_pre_o_proj` (F32 controlled) | 1.000000 | 1.000000 | ✓ |
| `L0.attn_out_pre_o_proj` (real F16 KV cache) | 0.962 | 0.83 | ≠ (F16 noise) |
| `model.norm` | 0.999997 | 0.999999 | ✓ |
| `post_projection` | 0.999995 | 0.999999 | ✓ |
| **logits (masked or plain)** | **0.999999** | **1.000000** | **✓** |

End-to-end speculative decoding via `llama-server` produces coherent text:

| Model | Backend | Acceptance rate |
|---|---|---|
| E2B-it-assistant | CPU + GPU | 0% (intrinsic — drafter weak on standalone prompts) |
| 31B-it-assistant | CPU | **8.3%** ("The capital of France is" → "Paris.\\n\\nThe capital of France is Paris...") |
| 31B-it-assistant | GPU (`-ngl 99`) | crashes during init — see `docs/31B_STATUS.md` |

See `docs/` for the design and per-bug post-mortems.

## Quick start

### 1. Convert the drafter to an MTP overlay GGUF

```bash
python convert_hf_to_gguf.py \
    ~/.cache/huggingface/hub/models--google--gemma-4-E2B-it-assistant/snapshots/<hash> \
    --outfile gemma-4-E2B-it-mtp.gguf --outtype f16
```

This produces a small (~150 MB for E2B) GGUF containing only the drafter's
4 transformer blocks plus `pre_projection`, `post_projection`, tied
`embed_tokens`, and the `masked_emb_centroids` + `masked_emb_token_ordering`
buffers used for the centroid-based output projection.

The base GGUF is the standard conversion of `google/gemma-4-E2B-it`.

### 2. Run llama-server with the overlay

```bash
build/bin/llama-server \
    -m gemma-4-E2B-it.gguf \
    -md gemma-4-E2B-it-mtp.gguf \
    --spec-type draft-mtp \
    -c 512
```

The server auto-detects that the `-md` file is an overlay (probed via
`llama_gguf_is_mtp_overlay()`), attaches it to the base model in-place via
`llama_model_load_mtp_overlay()`, and creates a draft context against the
same model with `ctx_type = LLAMA_CONTEXT_TYPE_MTP`.

### 3. Generate text

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "messages":[{"role":"user","content":"Explain quicksort in one sentence."}],
      "max_tokens": 60, "temperature": 0
    }'
```

## What the port does

### Loader (`llama_model_load_mtp_overlay`)

Attaches a Gemma4Assistant overlay GGUF to an already-loaded base model.
Validates KV metadata, type-checks every tensor against the expected
shape, and stows them in `model->mtp.*`. The overlay's `ggml_context`
lifetime transfers to `model->mtp.overlay_ctx`.

Optional `masked_embedding` tensors (`masked_emb_centroids`,
`masked_emb_token_ordering`) are loaded when
`hparams.mtp_use_ordered_embeddings` is set — these power the
centroid-based sparse output projection used by E2B and similar variants.

### MTP context (`LLAMA_CONTEXT_TYPE_MTP`)

A separate `llama_context` type that builds the drafter's forward graph
(`gemma4.cpp::graph_mtp`) instead of the base model's. The graph
consumes four runtime inputs bound by the driver:

| Binding name | Purpose |
|---|---|
| `mtp_h_input` | base's `last_hidden_state` at position `t-1` (recurrent input) |
| `mtp_shared_K_swa`, `mtp_shared_V_swa` | last sliding-attention layer's K/V from base, captured in F16 |
| `mtp_shared_K_full`, `mtp_shared_V_full` | last full-attention layer's K/V from base |
| `mtp_attn_mask` | F32 mask, `0` for valid K positions and `-inf` for zero-padded slots |

The drafter's cross-attention reads the bound K/V (no KV cache writes of
its own), and uses RoPE on `Q` only since the captured K is already
post-RoPE.

### Masked embedding (E2B / `use_ordered_embeddings`)

Variants like E2B-it-assistant don't compute logits as `lm_head(x)`.
Instead they:

1. project to a small "centroid" space (`centroids @ x` → `[2048]`)
2. pick the top-K centroids per position
3. gather each cluster's vocab indices from `token_ordering`
4. compute dot products only against those token embeddings
5. scatter the resulting `K*vocab_per_centroid` scores into a full vocab
   tensor, filling unselected positions with `min(selected) − 1.0`

The C++ implementation **outputs the sparse representation directly**
(`selected_logits` + `selected_indices` as debug taps from
`graph_mtp`) and **reconstructs the full-vocab logits host-side** in
`llama_context::process_ubatch`. This avoids a `[vocab=262144, n_tokens]`
intermediate tensor that crashed `sched_reserve()`'s memory planner. The
reconstructed logits are written into the standard logits buffer so
`llama_get_logits()` and the sampler stack just work.

### Per-token post-norm `h` API

The drafter expects the base model's **post-norm** `h` (output of
`model.output_norm`), not pre-norm. Two new public APIs:

```c
LLAMA_API const float * llama_get_last_hidden_state    (struct llama_context *);
LLAMA_API const float * llama_get_last_hidden_state_ith(struct llama_context *, int32_t i);
LLAMA_API const float * llama_get_h_pre_norm           (struct llama_context *);
LLAMA_API const float * llama_get_h_pre_norm_ith       (struct llama_context *, int32_t i);
```

The driver (`common/speculative.cpp::common_speculative_impl_draft_mtp`)
uses these to feed per-position `h` into `batch.embd` during batched
verification.

### CLI integration

`tools/server/server-context.cpp` detects when the `-md` path is an
overlay GGUF (via `llama_gguf_is_mtp_overlay()`) and routes it through
the overlay loader instead of `llama_model_load_from_file()`. The
`--spec-type draft-mtp` flag is unchanged from upstream — the
detection is automatic.

## What we found while porting

Each bug below was localized with a different test artifact in `tests/`
and `tools/`:

| # | Bug | How we found it | Fix |
|---|---|---|---|
| 1 | `mtp_h_input` was declared as a graph input but never bound | mini-spec generated nonsense even though K/V binding worked | added `t_last_hidden_state` capture + binding plumbing |
| 2 | Q@K scaling was `1/√d` (standard transformer); Gemma4 uses `1.0` | layer-by-layer diff harness, `L0.attn_out_pre_o_proj` cos=0.60 | `kq_scale = 1.0f` |
| 3 | zero-padded K rows polluted the softmax | same harness — found via mask experiment | new `mtp_attn_mask` runtime binding |
| 4 | `norm(attn + residual)` vs HF's `residual + norm(attn)` | post-attn norm order ambiguity | moved the norm before the add |
| 5 | FFN activation was `silu`; Gemma4 uses `gelu_pytorch_tanh` | `L0.out` cos=0.992 after fix #4 | `LLM_FFN_GELU` |
| 6 | E2B uses `masked_embedding`, not plain `lm_head` | `logits` cos=−0.73 with everything upstream at cos=1.0 | sparse-output design + host-side scatter |
| 7 | `llama_set_input_tensor` stores POINTERS, not copies | spec-test got NaN from a stack-scoped attn_mask vector | documented `!! LIFETIME HAZARD !!` in `llama.h` |
| 8 | `-md overlay.gguf` failed because llama-server expected standalone model | server load error | added `llama_gguf_is_mtp_overlay()` probe + routing |
| 9 | mask binding had wrong size for multi-token verify batches | log warned `bound 1024 B, expected 4096 B` | added `rebind_mtp_attn_mask()` called before every `llama_decode` |

## Test artifacts (`examples/gemma4-mtp/`)

### `tests/`

- `test-gemma4-mtp-validate.cpp` — structural validator: checks GGUF KV
  pairs + tensor types/shapes for an overlay file.
- `test-gemma4-mtp-e2e.cpp` — loads base + overlay, builds an MTP
  context, runs a smoke decode. Bare-minimum infrastructure proof.
- `test-gemma4-mtp-data-dependence.cpp` — runs MTP three times (A, B with
  different main-pass inputs, C identical to A). Confirms cross-attention
  is data-dependent and deterministic.
- `test-gemma4-mtp-full-flow.cpp` — main forward → K/V extraction →
  binding → MTP decode → real decoded tokens.
- `test-gemma4-mtp-mini-spec.cpp` — real prompt ("The quick brown fox")
  end-to-end through base + drafter, with sparse-output sampling for E2B.
- `test-gemma4-mtp-ref-diff.cpp` — reads `reference_inputs.npz` produced
  by `reference_probe.py`, runs the same inputs through C++, dumps
  activations to `acts_cpp.bin` for layer-by-layer diff.
- `test-gemma4-mtp-speculative.cpp` — full speculative-decoding loop
  (drafts + base verification + acceptance) without going through
  `common/speculative.cpp` — useful for isolating driver issues.

### `tools/`

- `reference_probe.py` — runs the HF Gemma4Assistant drafter on
  controlled seeded inputs, dumps inputs (.npz) + activations (.npz) for
  the C++ diff harness.
- `real_flow_probe.py` — runs the HF base + drafter on a real prompt
  ("The quick brown fox"), captures base's `last_hidden_state` and
  `shared_kv_states`, runs drafter. Reference for real-prompt diff.
- `diff_activations.py` — compares C++ acts (from `test-gemma4-mtp-ref-diff`)
  against the HF probe's activations, layer-by-layer.
- `diff_real_flow.py` — compares C++ acts (dumped by `mini-spec` when
  `MINI_SPEC_DUMP_DIR` env is set) against real_flow_probe's HF acts.
- `dump_inputs_to_bin.py` — converts `.npz` to flat `.bin` files for
  the C++ harness (no NPZ reader needed in C++).

## Known limitations

### F16 KV cache → 0.962 cos at `L0.attn_out_pre_o_proj` on real prompts

The base model's KV cache is stored in F16 (llama.cpp default). With
controlled F32 inputs (diff harness) we get cos=1.000000; with real F16
KV (mini-spec / real-flow) we get cos=0.962 at the cross-attention output.
The argmax recovers downstream — final logits still match HF — but
intermediate tap values drift slightly.

To eliminate this, create `ctx_tgt` with `type_k = type_v = GGML_TYPE_F32`.
Default F16 is fine for production.

### Drafter acceptance rate

For E2B on standalone non-chat prompts, draft acceptance is near 0%.
HF's own `base.generate(assistant_model=draft)` on the same inputs
shows similar misalignment (degenerate output on
`"The capital of France is"`). The drafter+base combo seems to need
in-distribution conversational context for the drafter to predict
tokens base will accept. This is intrinsic to the model, not the
port.

### `llama-cli` integration

The `-md` integration is currently only wired into `llama-server`.
`llama-cli` would need a similar branch — straightforward, just hasn't
been done.

## License & attribution

Same Apache 2.0 license as the rest of llama.cpp. The Gemma 4 models
themselves are governed by Google's Gemma Terms of Use. This port is
not affiliated with or endorsed by Google or the ggml-org team.

## Acknowledgements

- The upstream llama.cpp MTP scaffolding (Qwen35-style) introduced in
  [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) was the
  starting point.
- HuggingFace's `transformers.models.gemma4_assistant` was the reference
  implementation against which every layer was diffed.
