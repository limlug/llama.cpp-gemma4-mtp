# Root cause: `mtp_h_input` is never bound

## The bug

In `applied/cpp/gemma4.cpp` line 740-742, the MTP graph declares:

```cpp
inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, backbone_dim, n_tokens);
ggml_set_input(inp->embd);
ggml_set_name(inp->embd, "mtp_h_input");
```

This is supposed to receive the **previous backbone hidden state** `h` so the drafter can compute:

```
x = pre_projection( concat(h_in, token_embedding) )
```

But **no code anywhere actually binds `"mtp_h_input"`**:
- `applied/cpp/speculative.cpp` `common_speculative_impl_draft_mtp::process` never references it
- `tests/test-gemma4-mtp-mini-spec.cpp` and friends never reference it
- Only the K/V shared bindings (`mtp_shared_K_swa` etc.) are set

Consequence: the input buffer for `mtp_h_input` contains whatever the scheduler left there (zeros after a fresh `llama_clear_input_tensor_bindings`, garbage otherwise). The drafter then computes:

```
pre_projection( concat(0, e_t) )   ← wrong; should be concat(h_t, e_t)
```

This is consistent with the symptoms:
- Data-dependence test passed (K/V cross-attention is real and varies with main inputs)
- But mini-spec produced `"sizing"` then `<pad>` for "The quick brown fox" — because the input embedding pathway is broken regardless of which prompt you use, the drafter is operating on a half-zeroed input every time.

## The fix

Two halves:

**Capture side** — expose the base model's final hidden state from `ctx_tgt`:
1. Add `ggml_tensor * t_last_hidden_state` to `llm_graph_result` (parallels existing `t_shared_K_*`).
2. In the main `gemma4::graph::graph` constructor, after the last `model.norm`, capture via `ggml_dup` (just like K/V capture) and `ggml_build_forward_expand`.
3. Add `std::vector<float> last_hidden_state` buffer to `llama_context` and copy-out in the encode/decode `extract_shared_d` lambda.
4. Add `LLAMA_API const float * llama_get_last_hidden_state(llama_context *)` and `llama_get_last_hidden_state_size(llama_context *)`.

**Bind side** — wire `mtp_h_input` into the driver and test:
1. In `tests/test-gemma4-mtp-mini-spec.cpp`, after `llama_decode(ctx_tgt, b)`, grab the last-position slice from `llama_get_last_hidden_state(ctx_tgt)` and call `llama_set_input_tensor(ctx_dft, "mtp_h_input", h, backbone_dim * sizeof(float))`.
2. After each MTP step, grab `t_h_pre_norm` (already captured as `res->t_h_pre_norm` in `gemma4.cpp:867`) — needs the same capture/getter machinery: add `llama_get_h_pre_norm(ctx_dft)` — then rebind for the next step.
3. In `applied/cpp/speculative.cpp` `common_speculative_impl_draft_mtp::process`, do the same wiring for production driver.

## Test plan

1. Apply the patch.
2. Rebuild on dpl26.
3. Re-run `tools/repro_all.sh` — all 10 stages should still pass.
4. Re-run `tests/test-gemma4-mtp-mini-spec`. Expected new behavior: the MTP draft token at step 0 should be a much more plausible continuation of "The quick brown fox" than `"sizing"`.
5. Even if it isn't perfectly HF-matching yet, the change in output is the canary that the binding works. Numerical-match validation comes from the reference probe diff (next task).

## Verification on dpl26 (2026-05-20)

Rebuilt libllama with all patches, reran `tests/test-gemma4-mtp-mini-spec` against `gemma-4-E2B-it.gguf` + overlay.

**Before fix**: `"sizing"` → `<pad>` → `<pad>` (drafter ran with h=0)

**After fix**: `" 하겠습니다"` → `"륵"` → `"}}}$"` — output **changed**, confirming the binding now lands. Data-dependence test still passes (no regression in K/V pathway).

First-cut failure mode discovered: the `mtp_h_input` name wasn't in the `known[]` allowlist in `process_ubatch` (`llama-context.cpp` line ~1305). Fixed by adding a `t_inp_mtp_h_input` slot to `llm_graph_result`, setting it in `graph_mtp::graph_mtp`, and adding the entry to the allowlist. Three-file patch on top of the original change.

**Residual gap**: numerical fidelity to HF still off. Drift to Korean / closing-brace tokens suggests the captured `last_hidden_state` doesn't match what HF's `Gemma4Assistant` feeds the drafter. Plausible suspects:
- pre- vs post-final-RMSNorm for the captured `h`
- `inputs_embeds` scaling/normalization that HF applies before `pre_projection`
- concat-dim ordering (`ggml_concat dim=0` ↔ `torch.cat dim=-1`)

These are downstream of the now-fixed binding pathway. Next step is the layer-by-layer diff harness (probe expanded in `tools/reference_probe.py` already).

## Status

**Patches applied** (2026-05-20) — sources in this repo:
- `applied/cpp/llama-graph.h` — added `t_last_hidden_state` slot + getter
- `applied/cpp/llama-context.h` — added `last_hidden_state` / `h_pre_norm` buffers + accessors
- `applied/cpp/llama-context.cpp` — both extract paths now also copy `t_last_hidden_state` and `t_h_pre_norm`; C API functions `llama_get_last_hidden_state{,_size}` and `llama_get_h_pre_norm{,_size}` added
- `applied/cpp/llama.h` — public declarations for the four new APIs
- `applied/cpp/gemma4.cpp` (main `graph::graph`) — captures `cur` after `model.output_norm` via `ggml_dup` into `res->t_last_hidden_state`
- `tests/test-gemma4-mtp-mini-spec.cpp` — binds `mtp_h_input` to the captured `last_hidden_state` before MTP step 0, then rebinds to `h_pre_norm` after each step (recurrent feedback)

**Not yet patched**: `applied/cpp/speculative.cpp` already feeds h via `batch.embd` (using `llama_get_embeddings_pre_norm`). That path may or may not actually populate `mtp_h_input` correctly — needs verification once the test fix proves out. If the driver flow turns out to be wrong too, mirror the binding-style fix there.

## Why we missed it

- The shared K/V bindings were the obvious "drafter needs main-model context" hookup, and after wiring them and getting *something* out, we declared the structural path complete.
- The data-dependence test exercised the K/V pathway, not the `h_in` pathway, so it kept passing.
- The `mtp_h_input` declaration is in `gemma4.cpp` but the binding responsibility lives in the driver — there's no compile-time assertion that an `inp->embd` named input has a corresponding binding.

Could add a runtime warning if `llama_decode` runs on an MTP-typed context without `mtp_h_input` being bound. Worth doing.
