# Gemma 4 MTP port — change summary

Listed roughly in the order the work was done, with what changed and the
test that proved each step.

## Conversion (Python)

### `conversion/gemma.py`

- Added `Gemma4AssistantModel` class registered for the
  `Gemma4AssistantForCausalLM` HF architecture (`mtp_only = True`).
- `modify_tensors()` routes HF tensor names to canonical `mtp.*` names
  via `MODEL_TENSOR.MTP_*` enums. Disambiguates colliding names like
  `model.norm.weight` and `model.embed_tokens.weight` explicitly.
- Handles `masked_embedding.centroids.weight` and
  `masked_embedding.token_ordering` (cast to F32 since vocab fits
  losslessly).

### `gguf-py/gguf/constants.py`

- 17 new `MODEL_TENSOR` enums for `MTP_*` tensors (pre/post-proj, norm,
  embed_tokens, per-block weights, layer_scalar, plus the two
  masked-embedding tensors).
- New `Keys.MTP` sub-class with 15 metadata keys
  (`hidden_size`, `intermediate_size`, head counts, sliding window, eps,
  layer types array, RoPE params, masked-embedding params).
- `MODEL_TENSORS[MODEL_ARCH.GEMMA4]` extended with all MTP tensor names.

### `gguf-py/gguf/gguf_writer.py`

- 12 new `add_mtp_*()` helpers covering every `Keys.MTP.*` field.

## C++ runtime

### `include/llama.h`

- `LLAMA_API` declarations for:
  - `llama_gguf_is_mtp_overlay(path)` — probe whether a GGUF is an overlay.
  - `llama_model_load_mtp_overlay(model, path)` — attach an overlay to a model.
  - `llama_set_input_tensor(ctx, name, data, n_bytes)` — runtime input binding
    (with `!! LIFETIME HAZARD !!` warning since data is borrowed).
  - `llama_clear_input_tensor_bindings(ctx)`.
  - `llama_get_shared_kv_K_swa`/`V_swa`/`K_full`/`V_full` + `_size`.
  - `llama_get_last_hidden_state` + `_size` + `_ith` (per-token post-norm `h`).
  - `llama_get_h_pre_norm` + `_size` + `_ith` (drafter's recurrent output).
  - `llama_get_dbg_tap_count/name/data/size` (debug-tap mechanism).

### `src/llama-arch.{h,cpp}`

- 19 new `LLM_TENSOR_MTP_*` enums + their canonical name strings + op-info entries.
- 15 new `LLM_KV_MTP_*` enums + name strings.

### `src/llama-hparams.h`

- 13 new MTP-specific hparam fields (`mtp_n_embd`, head counts, head dims,
  sliding window, eps, layer types array, RoPE params,
  `mtp_use_ordered_embeddings`, `mtp_num_centroids`, `mtp_centroid_top_k`).
- `mtp_is_swa(il)` helper.

### `src/llama-graph.h`

- New `llm_graph_result` slots:
  - `t_last_hidden_state` — base's post-norm `h` capture.
  - `t_h_pre_norm` — drafter's `post_projection` output.
  - `t_shared_K_swa/V_swa/K_full/V_full` — base's main-pass K/V capture (per ubatch).
  - `t_inp_mtp_shared_K_swa/V_swa/K_full/V_full` — MTP-graph input tensors that
    the driver binds to via `llama_set_input_tensor`.
  - `t_inp_mtp_h_input` — drafter's `h_in` binding slot.
  - `t_inp_mtp_attn_mask` — cross-attention mask binding slot.
  - `t_dbg` — debug-tap registry (vector of `(name, ggml_tensor*)` pairs).
- Getters for each.

### `src/llama-model.{h,cpp}`

- New `llama_mtp_layer` struct (per-block drafter weights: `wq`, `attn_norm`,
  `attn_q_norm`, `wo`, `attn_post_norm`, `ffn_pre_norm`/`gate`/`up`/`down`/`post_norm`,
  optional `out_scale`).
- New `llama_mtp_block` struct (top-level: `pre_proj`, `post_proj`, `norm`,
  `embed_tokens`, `masked_emb_centroids`, `masked_emb_token_ordering`, layers,
  `overlay_ctx` for tensor data lifetime).
- `model->mtp` field added.
- `llama_gguf_is_mtp_overlay()` impl: opens GGUF in `no_alloc` mode, checks
  for `{arch}.mtp.hidden_size` AND absence of `{arch}.context_length`.
- `llama_model_load_mtp_overlay()` impl: opens overlay with `no_alloc=false`
  (eager CPU load), validates 13+ KV pairs + 48+ tensors with type+shape
  checks, transfers ggml context ownership to `model->mtp.overlay_ctx`.

### `src/llama-context.{h,cpp}`

- `last_hidden_state` and `h_pre_norm` host buffers (extracted per ubatch
  via `ggml_backend_tensor_get_async`).
- `shared_kv_K/V_swa/full` host buffers.
- `dbg_taps` map for debug-tap extraction.
- `input_tensor_bindings` machinery — applies stored
  `(name → data, n_bytes)` mappings to graph input tensors AFTER
  `set_inputs(ubatch)` but BEFORE forward. Allowlist of known binding
  names includes `mtp_shared_K_swa/V_swa/K_full/V_full`, `mtp_h_input`,
  `mtp_attn_mask`. Wrong-size bindings are rejected with a logged error.
- Public C API: `llama_get_*` accessors, `llama_set_input_tensor`,
  `llama_clear_input_tensor_bindings`.
- **Masked-logits host-side reconstruction** in `process_ubatch` decode
  path: when `model.arch == LLM_ARCH_GEMMA4 && hparams.mtp_use_ordered_embeddings`,
  reads `selected_logits` + `selected_indices` from `dbg_taps`, computes
  `mask_value = min(selected_logits) − 1.0`, scatters into the standard
  `logits.data` buffer. So `llama_get_logits()` returns HF-equivalent
  masked logits and the sampler stack just works.

### `src/models/gemma4.cpp`

- **Main graph (`graph::graph`)**: captures K/V from the LAST sliding-attn
  layer and LAST full-attn layer via `ggml_dup` of cache views post-`build_attn`.
  Captures post-`output_norm` `cur` as `res->t_last_hidden_state`.
- **`build_cross_attn_no_cache(q, k, v, mask, kq_scale, il)`**: no-cache
  attention used by graph_mtp. Accepts an attention mask and uses
  `kq_scale = 1.0f` (Gemma4 convention, NOT `1/√d`).
- **`graph_mtp` constructor**: builds the drafter's forward graph.
  - Declares `mtp_h_input`, `mtp_shared_K_swa/V_swa/K_full/V_full`,
    `mtp_attn_mask` as runtime-bindable inputs and stores pointers in
    `res->t_inp_mtp_*`.
  - Looks up next-token embedding from BASE's `model.tok_embd`.
  - `concat(h_in, tok_embd)` → `pre_proj` → 4 drafter blocks.
  - Each block: `input_layernorm` → `q_proj` → `q_norm` → RoPE on Q →
    `build_cross_attn_no_cache` → `o_proj` → `attn_post_norm(attn_out)` +
    residual → `ffn_pre_norm` → `build_ffn(LLM_FFN_GELU)` →
    `ffn_post_norm` → residual → `layer_scalar`.
  - Layer 0 emits debug taps for every intermediate.
  - Logits path: when `mtp_use_ordered_embeddings`, computes the sparse
    masked-embedding outputs (`top_k(centroids @ cur)` → `get_rows` on
    `token_ordering` → `get_rows` on `embed_tokens` → dot with `cur`) and
    emits them as debug taps (`selected_logits`, `selected_indices`).
    The scattered full-vocab tensor is reconstructed host-side in the
    context's extract path (see above).
  - Plain `lm_head` is ALSO computed and assigned to `res->t_logits` so
    `sched_reserve()` and downstream consumers have a valid tensor at any
    n_tokens — the host-side reconstruction overrides this for E2B-style
    variants.
- Loads new MTP hparams (`mtp_use_ordered_embeddings`, `num_centroids`,
  `centroid_top_k`) from KV metadata.

## Driver

### `common/speculative.cpp` (`common_speculative_impl_draft_mtp`)

- New `mtp_attn_mask_buf` member buffer (lifetime-safe storage for the
  `mtp_attn_mask` binding — fixes the silent-NaN bug where a
  stack-scoped vector caused dangling pointers).
- New `rebind_mtp_attn_mask(ctx_dft, n_batch_tokens)` helper, called
  before EVERY `llama_decode(ctx_dft, batch)` because the graph's mask
  tensor shape is `[n_ctx, n_tokens]` and the binding API enforces exact
  size match (verify and draft decodes have different n_tokens).
- `process()`: binds K/V from ctx_tgt, computes per-token `kv_len`
  (saved in `mtp_kv_len`), populates `batch.embd` with
  `llama_get_last_hidden_state_ith(ctx_tgt, k)` (post-norm — was
  previously pre-norm via `llama_get_embeddings_pre_norm`, which doesn't
  match HF).
- `draft()`: uses `llama_get_h_pre_norm_ith(ctx_dft, i_batch)` for the
  drafter's recurrent feedback (with a fallback to the legacy pre-norm
  getter for non-Gemma4 MTP variants).

## CLI / server

### `tools/server/server-context.cpp`

- New branch in `load_model()`: when `--spec-type draft-mtp` is set and
  the `-md` path is an overlay GGUF (detected via
  `llama_gguf_is_mtp_overlay()`), attach the overlay to `model_tgt` via
  `llama_model_load_mtp_overlay()` and create `ctx_dft` from the SAME
  model with `ctx_type = LLAMA_CONTEXT_TYPE_MTP`. Skips the broken
  `llama_model_load_from_file()` path that errored with
  `"key not found: gemma4.context_length"`.

## Tests + diagnostic tools (`examples/gemma4-mtp/`)

Eight C++ tests in `tests/` and five Python tools in `tools/`. See
`README.md` for the full inventory and what each one proves. The
`tools/reference_probe.py` + `tools/diff_activations.py` pair was
the critical instrument that localized each of the nine numerical
bugs to a specific layer.
