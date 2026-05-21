# K/V Provenance Fix — design note

## The issue

Current capture in `src/models/gemma4.cpp::graph::graph` is:

```cpp
if (il == last_kv_swa) {
    res->t_shared_K_swa = Kcur;   // ⚠ post-RoPE per-ubatch slice
    res->t_shared_V_swa = Vcur;
}
```

`Kcur` and `Vcur` are the per-ubatch K and V — they contain only the K/V for the tokens being decoded in THIS forward call, not the full accumulated context.

The HF reference (`modeling_gemma4.py`) sets `shared_kv_states[layer_type]` from the last-attention-layer's K/V **after the cache update**, so it's the full accumulated K/V at that layer (one entry per token in the full context so far).

For one-shot prefill with `n_tokens == kv_len`, our current capture is correct. For autoregressive decoding (one token per ubatch), it's wrong — the drafter ends up attending to a single K/V row instead of the full context.

## The fix

Use the KV cache's view API to get full-context K/V instead of the per-ubatch slice.

### Location

`src/models/gemma4.cpp::llama_model_gemma4::graph::graph`, inside the per-layer attention block, right where we currently do the capture.

### Required changes

1. **Get a handle to the KV cache from `inp_attn`.** The `build_attn_inp_kv_iswa()` call returns an `llm_graph_input_attn_kv_iswa *` which holds the KV cache pointer internally.

2. **Use `kv.get_k(ctx0, il, n_kv, sinfo)` and `kv.get_v(...)` to create view tensors** that span the entire accumulated K/V at the given layer index. `n_kv` is the current cache occupancy; `sinfo` is the slot info.

3. **Replace the capture lines:**

```cpp
// BEFORE
if (il == last_kv_swa) {
    res->t_shared_K_swa = Kcur;
    res->t_shared_V_swa = Vcur;
}

// AFTER
if (il == last_kv_swa) {
    auto * kv_swa = inp_attn->get_kv_swa();        // or equivalent accessor
    const auto & sinfo = inp_attn->get_sinfo_swa();
    const uint32_t n_kv = kv_swa->get_n_kv(sinfo);
    res->t_shared_K_swa = kv_swa->get_k(ctx0, il, n_kv, sinfo);
    res->t_shared_V_swa = kv_swa->get_v(ctx0, il, n_kv, sinfo);
}
```

(Names may need adjustment; the `llm_graph_input_attn_kv_iswa` struct's
exact accessor names should be checked at implementation time.)

### Shape consequences

The captured tensors will now have shape `[head_dim, n_head_kv, n_kv]` where `n_kv` is the full accumulated context length, not the per-ubatch token count. This matches the HF reference shape exactly.

The downstream extraction in `llama-context.cpp` and the host vectors already handle variable size — `extract_shared()` does `buf.resize(n_bytes / sizeof(float))` lazily, so no other code change is needed.

### Memory implications

For long contexts, the host-side vectors will be large:
- 256 (head_dim) × 16 (n_head_kv) × kv_len × 4 bytes = 16 KB × kv_len per layer per (K or V)
- At kv_len = 32K: ~500 MB total for the four shared K/V buffers

This is significant but tolerable. An optimization would keep the data device-resident and only copy to host once per accepted-token boundary (since the drafter runs N steps between captures).

## Validation plan

Once implemented, the existing `reference_activations.json` (argmax = 89004) can verify correctness. The fake input we used in the probe simulates the "full kv_len = 8" case which exercises the multi-position attention; the C++ should match.

## Why this can't be done before there's a base GGUF

Without the ability to actually run inference end-to-end, any change here is speculative. The current per-ubatch capture is structurally compatible with the rest of the pipeline (the buffers size correctly, the API path works), so the fix is purely about **what data** is in those buffers. Getting that wrong without a test loop means silent numerical errors that only show up when we finally have a base model to run.

## Effort estimate

Once a base GGUF is available: ~1 day of work (read KV cache API thoroughly, make the swap in graph::graph, run a small inference to verify the cache-view tensors have the right shape and content).
