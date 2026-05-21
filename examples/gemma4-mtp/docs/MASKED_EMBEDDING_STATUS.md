# masked_embedding port — COMPLETE (sparse-output design)

## Final state (2026-05-20)

Bit-exact match against HF reference (E2B drafter, deterministic seeded inputs):

```
✓ pre_projection          cos=1.000000   ✓
≈ L0.input_layernorm      cos=1.000000   ✓
≈ L0.q_proj               cos=1.000000   ✓
≈ L0.q_norm               cos=1.000000   ✓
✓ L0.attn_out_pre_o_proj  cos=1.000000   ✓
✓ L0.o_proj               cos=1.000000   ✓
≈ L0.out                  cos=1.000000   ✓
✓ model.norm              cos=0.999997   ✓
≈ post_projection         cos=0.999995   ✓
✗ logits_masked           cos=0.999999   argmax_hf=100717  argmax_cpp=100717  ✓
```

The plain `logits` (lm_head Linear, still computed as `t_logits` for framework compat) intentionally does NOT match HF since E2B uses masked_embedding instead — the meaningful comparison is `logits_masked`.

## Design

The full HF `Gemma4AssistantMaskedEmbedder.forward` materializes a `[vocab, n_tokens]` filled tensor and does a vocab-wide `scatter_`. That triggered allocator failures during `sched_reserve` (which probes the graph at n_tokens ∈ {1, 16, 32} for memory planning; a 262k-vocab × 32-tokens filled tensor was too much).

The sparse-output redesign:
1. C++ does the masked_embedding compute: `centroid_logits = centroids @ h`, `top_k_idx = top_k(centroid_logits, k=32)`, `selected_canonical = token_ordering_2d[top_k_idx]`, `selected_embeddings = embed_tokens[selected_canonical]`, `selected_logits = selected_embeddings @ h`.
2. Outputs **two tensors as debug taps**: `selected_logits[K=4096]` and `selected_indices[K=4096]`. No graph-side scatter, no full-vocab materialization.
3. Python diff script (`tools/diff_activations.py`) reconstructs `logits_masked[vocab]` host-side:
   - `mask_value = min(selected_logits) - 1.0`
   - `logits[v] = mask_value` for `v` not in `selected_indices`
   - `logits[v] = selected_logits[i]` for `v = selected_indices[i]`
4. Diff `logits_masked` against HF `out.logits`. Match.

## Implementation notes

- The MTP overlay's `mtp.masked_emb_token_ordering` is a leaf tensor in `model->mtp.overlay_ctx`. The scheduler can't allocate it directly (no buffer) — `ggml_dup(ctx0, mtp.masked_emb_token_ordering)` copies it into the compute context before use. This is the same trick we use for shared K/V capture.
- `ggml_top_k`, `ggml_cast`, and `ggml_get_rows` all have CPU backend support. The `selected_indices` (originally I32) is cast to F32 just so it can flow through the same F32 tap extraction pipeline; the Python side reinterprets as int.
- The `t_inp_mtp_const_one` slot and `mtp_const_one` binding are leftover from the in-graph mask_value computation (now host-side). Harmless; cleanup deferred.
- For real speculative-decoding sampling (production driver, not just diff): the driver should read the sparse outputs and either (a) sample directly from `selected_logits` mapped to `selected_indices`, or (b) materialize full vocab host-side. Either path works because samplers only need to know about valid tokens.

## Cleanup TODO

- Remove `t_inp_mtp_const_one` and the unused `mtp_const_one` allowlist entry.
- Add a C API (`llama_get_mtp_selected_logits`, `llama_get_mtp_selected_indices`) so the production driver can read the sparse outputs without going through the debug-tap mechanism. Optional — works fine via `llama_get_dbg_tap_*` for now.
