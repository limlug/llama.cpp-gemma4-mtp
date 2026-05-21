# Gemma4 MTP Design — v2 (post-reference-probe)

**Supersedes the original `DESIGN.md` on every point that conflicts.** Created
after reading `modeling_gemma4_assistant.py` directly and running a numerical
forward probe (see `tools/reference_probe.py` and `tools/reference_activations.json`).

## What we got wrong in v1

| v1 claim | Reality |
|---|---|
| Drafter's `embed_tokens [262144, 1024]` is **vestigial / unused** | **It IS used** — `lm_head` is **tied** to it via `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`. Logits come from `lm_head @ last_hidden_state_at_1024dim`. |
| Logits via main model's tied lm_head over 5376 dim | Logits via drafter's own tied head at **1024 dim**. The main model is uninvolved. |
| Attention K/V derivation unknown ("attention_k_eq_v: true") | K/V are **not computed** at all — they come from `shared_kv_states`, a runtime dict containing K and V tensors from the **main model's last attention layer of each type** (one for `full_attention`, one for `sliding_attention`). |
| `post_projection` output is what's captured as `t_h_pre_norm` for next step | Correct, but the reason is precisely because: `post_projection(last_hidden_state)` produces a **backbone-dim** state that becomes the next step's "h" half of the concat input. |
| The drafter doesn't consume `input_ids` (correct) | Confirmed: `input_ids: torch.Tensor | None = None,  # Not actually used, only kept in signature to be ignored`. |

## Actual architecture (verified)

```text
main model decode (per ubatch):
  → for the LAST sliding-attention layer of the main model:
        emit K_swa  [batch, main_kv_heads, kv_len, 256]
        emit V_swa  [batch, main_kv_heads, kv_len, 256]
  → for the LAST full-attention layer of the main model:
        emit K_full [batch, main_kv_heads, kv_len, 512]
        emit V_full [batch, main_kv_heads, kv_len, 512]
  → emit last hidden state h_main [batch, seq, 5376]

draft step (called N times per accepted token, loop body):
  inputs:
    h        : [batch, seq, 5376]   # h_main on step 0, post_proj(prev draft) on later steps
    next_tok : [batch, seq]         # last sampled token id
    K_swa/V_swa, K_full/V_full      # CAPTURED ONCE from main pass, reused across all N draft steps
  body:
    e = main_model.embed_tokens[next_tok]      # [batch, seq, 5376]
    x = pre_projection(concat(h, e))           # [batch, seq, 1024]    (10752 → 1024)
    for layer in 4 drafter layers (3 swa, 1 full):
        x = drafter_attn(Q=q_proj(x), K=shared_K_for_layer_type, V=shared_V_for_layer_type)
        x = mlp(x)
    x = norm(x)                                # [batch, seq, 1024]
    next_h    = post_projection(x)             # [batch, seq, 5376]   ← becomes next iter's h
    logits    = lm_head(x)                     # [batch, seq, 262144] using TIED 1024-dim head
    sample → next_tok for the next iteration
```

Note the **two distinct output paths from `x` (post-final-norm):**
- One goes through `post_projection` → 5376-dim → becomes next-step input `h`
- The other goes through `lm_head` (tied to drafter's own 1024-dim `embed_tokens`) → 262144-dim logits

## Critical architectural difference from Qwen35 MTP

Qwen35 MTP **computes K and V locally** inside the MTP block from its own `wk`/`wv`. Gemma4 MTP **inherits K and V from the main model's final attention layers** — there are literally no K/V projection weights in the drafter checkpoint.

This means the llama.cpp PR #22673 framework, designed for Qwen-style "extra decoder block at end of stack" MTP, cannot handle Gemma4 MTP **without additional driver-side plumbing**:

1. **Main pass must export `shared_kv_states`-equivalent tensors** — capture K and V from the last full-attention layer and the last sliding-attention layer.
2. **MTP context must consume them** — the MTP graph needs four extra ggml inputs (K_swa, V_swa, K_full, V_full).
3. **Each drafter layer's attention is cross-attention** — Q from layer input, K and V from the appropriate shared tensor (selected by layer type).

## Updated effort estimate

| Phase | Original v1 estimate | Revised v2 |
|---|---|---|
| Conversion script | 2–3 days | **0.5 day** (already done; minor tweak: keep `mtp.embed_tokens` always required, not optional) |
| GGUF schema | included above | unchanged |
| C++ struct + hparams | 1–2 days | 1 day |
| `graph_mtp` (with cross-attention) | 3–4 days | **3–4 days** (similar; now we know what to write) |
| Main-pass `shared_kv_states` export | not in v1 | **2–3 days** (NEW — touches `llama-context.cpp`, requires capturing K/V from chosen layers and exposing as additional outputs alongside `t_h_pre_norm`) |
| Driver loop changes in `common/speculative.cpp` | included | **2 days** (NEW — capture/pass `shared_kv_states` across MTP steps; PR #22673's driver assumes K/V are computed inside the MTP graph) |
| Test, debug, upstream | 5–8 days | 5–8 days |
| **Total** | 2–3 weeks | **~3 weeks** (similar; the new driver work is offset by no more "unknown semantics" risk) |

## Updated conversion script delta

The only change vs. patch 01:

```diff
-        # >>> Gemma4-style MTP overlay (top-level — no per-block index)
-        MODEL_TENSOR.MTP_EMBED_TOKENS: (
-            "model.embed_tokens",
-            "mtp.embed_tokens",
-        ),
+        # >>> Gemma4-style MTP overlay (top-level — no per-block index)
+        # mtp.embed_tokens is REQUIRED — it's the tied lm_head for the drafter's
+        # logits production at mtp_hidden (1024) dim.
+        MODEL_TENSOR.MTP_EMBED_TOKENS: (
+            "model.embed_tokens",
+            "mtp.embed_tokens",
+        ),
```

(No code change — the entry was already there. Just a doc correction.)

In patch 09's tensor loading, change:

```diff
-    mtp.embed_tokens = create_tensor(tn(LLM_TENSOR_MTP_EMBED_TOKENS, "weight"),
-                                     {mtp_n_embd, n_vocab},          TENSOR_NOT_REQUIRED);
+    mtp.embed_tokens = create_tensor(tn(LLM_TENSOR_MTP_EMBED_TOKENS, "weight"),
+                                     {mtp_n_embd, n_vocab},          0);  // required
```

## Updated graph_mtp pseudocode

```cpp
llama_model_gemma4::graph_mtp::graph_mtp(
        const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params), model(model)
{
    const auto & mtp = model.mtp;
    GGML_ASSERT(mtp.pre_proj && mtp.post_proj && mtp.norm && mtp.embed_tokens);

    const uint32_t n_predict     = hparams.nextn_predict_layers;
    const uint32_t mtp_n_embd    = hparams.mtp_n_embd;       // 1024
    const uint32_t backbone_dim  = hparams.n_embd;            // 5376
    const uint32_t mtp_n_head    = hparams.mtp_n_head;        // 32
    const uint32_t mtp_n_head_kv = hparams.mtp_n_head_kv;     // 16
    const uint32_t head_dim_swa  = hparams.mtp_n_embd_head_k; // 256
    const uint32_t head_dim_full = hparams.mtp_global_head_dim; // 512

    // --- inputs: 4 tensors per step ---
    //   inp->tokens       : [n_tokens]       (next token id; we look up its embed via model.tok_embd)
    //   inp->embd         : [backbone, n_tokens]  (recurrent h — backbone-dim)
    //   inp->shared_K_swa : [head_dim_swa, mtp_n_head_kv, kv_len, n_seqs]
    //   inp->shared_V_swa : same shape
    //   inp->shared_K_full: [head_dim_full, mtp_n_head_kv, kv_len, n_seqs]
    //   inp->shared_V_full: same shape

    auto inp = std::make_unique<llm_graph_input_mtp_gemma4>();
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->h_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, backbone_dim, n_tokens);
    ggml_set_input(inp->h_in);

    inp->shared_K_swa  = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_dim_swa,  mtp_n_head_kv, kv_len, n_seqs);
    inp->shared_V_swa  = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_dim_swa,  mtp_n_head_kv, kv_len, n_seqs);
    inp->shared_K_full = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_dim_full, mtp_n_head_kv, kv_len, n_seqs);
    inp->shared_V_full = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_dim_full, mtp_n_head_kv, kv_len, n_seqs);
    ggml_set_input(inp->shared_K_swa);
    ggml_set_input(inp->shared_V_swa);
    ggml_set_input(inp->shared_K_full);
    ggml_set_input(inp->shared_V_full);

    // Look up token embedding from BACKBONE's tok_embd → [backbone, n_tokens]
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);

    res->add_input(std::move(inp));

    // --- pre_projection ---
    ggml_tensor * concat = ggml_concat(ctx0, inp->h_in, tok_embd, /*dim=*/0);  // [2*backbone, n_tokens]
    ggml_tensor * cur    = build_lora_mm(mtp.pre_proj, concat);                // [mtp_n_embd, n_tokens]

    // --- 4 drafter blocks (3 sliding + 1 full) ---
    ggml_tensor * inp_pos = build_inp_pos();

    for (uint32_t il = 0; il < n_predict; ++il) {
        const auto & L      = mtp.layers[il];
        const bool is_swa   = hparams.mtp_is_swa(il);
        ggml_tensor * shared_K = is_swa ? inp->shared_K_swa : inp->shared_K_full;
        ggml_tensor * shared_V = is_swa ? inp->shared_V_swa : inp->shared_V_full;
        const uint32_t hd     = is_swa ? head_dim_swa : head_dim_full;

        // attention block
        ggml_tensor * residual = cur;
        cur = build_norm(cur, L.attn_norm, nullptr, LLM_NORM_RMS, il);

        ggml_tensor * Qcur = build_lora_mm(L.wq, cur);
        Qcur = ggml_reshape_3d(ctx0, Qcur, hd, mtp_n_head, n_tokens);
        Qcur = build_norm(Qcur, L.attn_q_norm, nullptr, LLM_NORM_RMS, il);

        // RoPE
        const float freq_base = is_swa
                ? hparams.mtp_rope_sliding_theta
                : hparams.mtp_rope_full_theta;
        const uint32_t n_rot = is_swa
                ? hd
                : (uint32_t)(hd * hparams.mtp_rope_full_partial_factor);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, /*rope_freqs=*/nullptr, n_rot,
                             /*rope_type=*/0, /*n_ctx_orig=*/0, freq_base,
                             1.0f, 0.0f, 1.0f, 0, 0);

        // Cross-attention: Q from drafter, K/V from main model.
        // shared_K/V are already in attention-ready layout from the main pass.
        // Use build_attn directly (no KV cache writes — these are read-only inputs).
        cur = build_attn_cross(L.wo, Qcur, shared_K, shared_V,
                               /*scale=*/1.0f / sqrtf((float) hd), il);

        cur = ggml_add(ctx0, cur, residual);
        cur = build_norm(cur, L.attn_post_norm, nullptr, LLM_NORM_RMS, il);

        // FFN with dual pre/post norms (Gemma4 quirk)
        ggml_tensor * ffn_in  = build_norm(cur, L.ffn_pre_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * ffn_out = build_ffn(ffn_in,
                L.ffn_up,   nullptr, nullptr,
                L.ffn_gate, nullptr, nullptr,
                L.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        ffn_out = build_norm(ffn_out, L.ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        cur = ggml_add(ctx0, cur, ffn_out);

        if (L.out_scale) {
            cur = ggml_mul(ctx0, cur, L.out_scale);
        }
    }

    // --- final norm ---
    cur = build_norm(cur, mtp.norm, nullptr, LLM_NORM_RMS, -1);
    // cur is now [mtp_n_embd, n_tokens] = [1024, n_tokens]

    // --- TWO outputs from cur ---

    // 1. Recurrent state: post_projection → [backbone, n_tokens]
    ggml_tensor * h_next = build_lora_mm(mtp.post_proj, cur);
    cb(h_next, "h_pre_norm", -1);     // misnamed but it's what the driver expects
    res->t_h_pre_norm = h_next;

    // 2. Logits via tied lm_head — drafter's own embed_tokens [vocab, 1024]
    ggml_tensor * logits = build_lora_mm(mtp.embed_tokens, cur);
    cb(logits, "result_output", -1);
    res->t_logits = logits;

    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, h_next);
}
```

`build_attn_cross` is a helper that does Q · K^T · V with no KV-cache write
(we're consuming externally provided K/V). This may already exist in llama.cpp
as a primitive; if not, it's a thin wrapper around `ggml_mul_mat` and the
attention softmax.

## Driver changes — `common/speculative.cpp` and `llama-context.cpp`

Per PR #22673, the existing MTP driver:
1. Runs main pass → captures `t_h_pre_norm` into `embd_pre_norm` buffer
2. Builds an MTP context (`LLAMA_CONTEXT_TYPE_MTP`)
3. For each draft step:
   - Sets MTP input `embd` = previous step's hidden, `tokens` = previous sampled token
   - Runs MTP forward → reads logits, samples
   - Captures THIS step's `t_h_pre_norm` for next iteration

For Gemma4 we need to **additionally** capture from the main pass:
- `shared_K_swa`, `shared_V_swa` (from main's last sliding-attn layer)
- `shared_K_full`, `shared_V_full` (from main's last full-attn layer)

These are captured **once per ubatch** (they don't change across MTP steps within
the same accepted-token boundary) and re-bound as inputs on every draft step.

Implementation sketch:

```cpp
// in llama_model_gemma4::graph::graph (main pass):
//   At the LAST sliding-attention layer:
//       cb(Kcur, "shared_K_swa", il);
//       res->t_shared_K_swa = Kcur;
//       res->t_shared_V_swa = Vcur;
//   Same for the LAST full-attention layer:
//       res->t_shared_K_full = Kcur;
//       res->t_shared_V_full = Vcur;

// in llama-context.cpp main-pass post-processing:
//   if (cparams.embeddings_pre_norm && t_shared_K_swa) {
//       ggml_backend_tensor_get_async(... res->t_shared_K_swa ... shared_kv_swa_buf ...);
//       ... same for V_swa, K_full, V_full
//   }

// in common/speculative.cpp draft loop:
//   for each MTP step:
//       set inp->shared_K_swa = shared_kv_swa_buf[K]
//       set inp->shared_V_swa = shared_kv_swa_buf[V]
//       set inp->shared_K_full = shared_kv_full_buf[K]
//       set inp->shared_V_full = shared_kv_full_buf[V]
//       (plus the existing h, tokens setup)
//       run MTP forward
```

This needs a new `llama_context_type` capability beyond what PR #22673 added,
OR can piggyback on `LLAMA_CONTEXT_TYPE_MTP` with arch-specific input handling.
The latter is cleaner since the driver doesn't need to know about the inputs;
they're set as named ggml inputs and the MTP graph's `add_input` mechanism
handles the binding.

## Test fixture

`tools/reference_activations.json` is now a captured ground-truth fixture:
- 16-element samples of every important intermediate (`pre_projection`, every
  layer's `q_proj`, `q_norm`, `o_proj`, layer output, `model.norm`,
  `post_projection`, `lm_head` logits)
- Full output shapes and the argmax of logits[0, 0] = **89004**

A C++ unit test should load this JSON, run `graph_mtp` with the same RNG-derived
inputs, and compare each intermediate within a tolerance (e.g., 1e-3 for fp32,
1e-2 for fp16).
