#include "models.h"
#include "../llama-kv-cache-iswa.h"
#include "../llama-kv-cache.h"
#include <cstdio>

void llama_model_gemma4::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);

    hparams.n_layer_kv_from_start = hparams.n_layer - (int32_t)n_kv_shared_layers;
    hparams.f_attention_scale     = 1.0f; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    switch (hparams.n_layer) {
        case 30: type = LLM_TYPE_26B_A4B; break;
        case 35: type = LLM_TYPE_E2B; break;
        case 42: type = LLM_TYPE_E4B; break;
        case 60: type = LLM_TYPE_31B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    // -- MTP overlay metadata (optional; present when the GGUF was built
    //    from -it-assistant via the new conversion path). --
    uint32_t mtp_predict_layers = 0;
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, mtp_predict_layers, false);
    if (mtp_predict_layers > 0) {
        hparams.nextn_predict_layers = mtp_predict_layers;
        ml.get_key(LLM_KV_MTP_HIDDEN_SIZE,        hparams.mtp_n_embd,          true);
        ml.get_key(LLM_KV_MTP_INTERMEDIATE_SIZE,  hparams.mtp_n_ff,            true);
        ml.get_key(LLM_KV_MTP_N_HEAD,             hparams.mtp_n_head,          true);
        ml.get_key(LLM_KV_MTP_N_HEAD_KV,          hparams.mtp_n_head_kv,       true);
        // Optional: separate KV head count for full-attention layers.
        // Defaults to mtp_n_head_kv when the key is absent (E2B-style).
        hparams.mtp_global_n_head_kv = hparams.mtp_n_head_kv;
        ml.get_key(LLM_KV_MTP_GLOBAL_N_HEAD_KV,   hparams.mtp_global_n_head_kv, false);
        ml.get_key(LLM_KV_MTP_HEAD_DIM,           hparams.mtp_n_embd_head_k,   true);
        ml.get_key(LLM_KV_MTP_GLOBAL_HEAD_DIM,    hparams.mtp_global_head_dim, true);
        ml.get_key(LLM_KV_MTP_SLIDING_WINDOW,     hparams.mtp_sliding_window,  false);
        ml.get_key(LLM_KV_MTP_LAYER_NORM_EPS,     hparams.mtp_f_norm_rms_eps,  false);

        // layer types array (0=sliding, 1=full); stored as INT32 array
        std::vector<int32_t> layer_types;
        ml.get_arr(LLM_KV_MTP_LAYER_TYPES, layer_types, true);
        GGML_ASSERT(layer_types.size() == hparams.nextn_predict_layers && "mtp.layer_types size mismatch");
        for (size_t i = 0; i < layer_types.size(); ++i) {
            hparams.mtp_layer_types[i] = (uint8_t) layer_types[i];
        }

        uint32_t rope_full_e6 = 0, rope_swa_e3 = 0;
        ml.get_key(LLM_KV_MTP_ROPE_FULL_THETA_E6,         rope_full_e6,                          false);
        ml.get_key(LLM_KV_MTP_ROPE_SLIDING_THETA_E3,      rope_swa_e3,                           false);
        ml.get_key(LLM_KV_MTP_ROPE_FULL_PARTIAL_FACTOR,   hparams.mtp_rope_full_partial_factor,  false);
        if (rope_full_e6 > 0) hparams.mtp_rope_full_theta    = (float) rope_full_e6 * 1e6f;
        if (rope_swa_e3  > 0) hparams.mtp_rope_sliding_theta = (float) rope_swa_e3  * 1e3f;
    }
}

void llama_model_gemma4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const uint32_t n_embd_per_layer = hparams.n_embd_per_layer;
    const int64_t  n_ff_exp         = hparams.n_ff_exp;

    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k_swa == n_embd_head_v_swa");
    }

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    if (n_embd_per_layer > 0) {
        per_layer_tok_embd   = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),    {n_embd_per_layer * n_layer, n_vocab}, 0);
        per_layer_model_proj = create_tensor(tn(LLM_TENSOR_PER_LAYER_MODEL_PROJ, "weight", 0), {n_embd, n_embd_per_layer * n_layer}, 0);
        per_layer_proj_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ_NORM,  "weight", 0), {n_embd_per_layer}, 0);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_embd_k    = hparams.n_embd_k_gqa(i);
        const int64_t n_embd_v    = hparams.n_embd_v_gqa(i);
        const int     kv_flags    = hparams.has_kv(i) ? 0 : TENSOR_NOT_REQUIRED;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k}, kv_flags);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v}, TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head * n_head, n_embd}, 0);

        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);
        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head}, kv_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.out_scale = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, TENSOR_NOT_REQUIRED);

        if (!hparams.is_swa(i)) {
            // full_attention layers use rope_freqs for proportional rope
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head/2}, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }

        // handle use_double_wide_mlp
        int64_t n_ff_cur = hparams.n_ff(i);

        // for expert layers, we use normal FFN as shared expert (same as python code)
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_cur, n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        // MoE router
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        bool has_expert = layer.ffn_gate_inp != nullptr;

        // norm
        if (has_expert) {
            layer.ffn_gate_inp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "scale", i), {n_embd}, 0);

            layer.ffn_pre_norm_2  = create_tensor(tn(LLM_TENSOR_FFN_PRE_NORM_2,  "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_1 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_1, "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_2 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_2, "weight", i), {n_embd}, 0);

            // MoE FFN
            layer.ffn_gate_up_exps  = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS,  "weight", i), {n_embd, n_ff_exp * 2, n_expert}, TENSOR_NOT_REQUIRED);

            if (layer.ffn_gate_up_exps == nullptr) {
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
                layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            }

            layer.ffn_down_exps     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,     "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

            // per-expert scale will be loaded as down_exps_s at the end of the current switch case
        }

        // per-layer embeddings
        if (n_embd_per_layer > 0) {
            layer.per_layer_inp_gate   = create_tensor(tn(LLM_TENSOR_PER_LAYER_INP_GATE,  "weight", i), {n_embd, n_embd_per_layer}, 0);
            layer.per_layer_proj       = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ,      "weight", i), {n_embd_per_layer, n_embd}, 0);
            layer.per_layer_post_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_POST_NORM, "weight", i), {n_embd}, 0);
        }
    }

    // -- MTP overlay tensors (loaded only when present) --
    if (hparams.nextn_predict_layers > 0 && hparams.mtp_n_embd > 0) {
        const int64_t mtp_n_embd  = hparams.mtp_n_embd;
        const int64_t mtp_n_ff    = hparams.mtp_n_ff;
        const int64_t n_predict   = hparams.nextn_predict_layers;
        const int64_t backbone_dim = n_embd;

        mtp.pre_proj     = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJ,     "weight"),
                                         {2 * backbone_dim, mtp_n_embd}, 0);
        mtp.post_proj    = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJ,    "weight"),
                                         {mtp_n_embd, backbone_dim},     0);
        mtp.norm         = create_tensor(tn(LLM_TENSOR_MTP_NORM,         "weight"),
                                         {mtp_n_embd},                   0);
        // Tied lm_head for the drafter's logits at mtp_hidden dim — REQUIRED.
        mtp.embed_tokens = create_tensor(tn(LLM_TENSOR_MTP_EMBED_TOKENS, "weight"),
                                         {mtp_n_embd, n_vocab},          0);

        mtp.layers.resize(n_predict);

        for (int64_t il = 0; il < n_predict; ++il) {
            auto & L = mtp.layers[il];

            const bool is_swa = hparams.mtp_is_swa((uint32_t)il);
            const int64_t head_dim = is_swa ? hparams.mtp_n_embd_head_k : hparams.mtp_global_head_dim;
            const int64_t q_dim    = head_dim * hparams.mtp_n_head;

            L.attn_norm      = create_tensor(tn(LLM_TENSOR_MTP_ATTN_NORM,      "weight", (int)il), {mtp_n_embd}, 0);
            L.attn_post_norm = create_tensor(tn(LLM_TENSOR_MTP_ATTN_POST_NORM, "weight", (int)il), {mtp_n_embd}, 0);
            L.ffn_pre_norm   = create_tensor(tn(LLM_TENSOR_MTP_FFN_PRE_NORM,   "weight", (int)il), {mtp_n_embd}, 0);
            L.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_MTP_FFN_POST_NORM,  "weight", (int)il), {mtp_n_embd}, 0);

            L.wq          = create_tensor(tn(LLM_TENSOR_MTP_ATTN_Q,      "weight", (int)il), {mtp_n_embd, q_dim}, 0);
            L.attn_q_norm = create_tensor(tn(LLM_TENSOR_MTP_ATTN_Q_NORM, "weight", (int)il), {head_dim},          0);
            L.wo          = create_tensor(tn(LLM_TENSOR_MTP_ATTN_OUTPUT, "weight", (int)il), {q_dim, mtp_n_embd}, 0);

            L.ffn_gate = create_tensor(tn(LLM_TENSOR_MTP_FFN_GATE, "weight", (int)il), {mtp_n_embd, mtp_n_ff}, 0);
            L.ffn_up   = create_tensor(tn(LLM_TENSOR_MTP_FFN_UP,   "weight", (int)il), {mtp_n_embd, mtp_n_ff}, 0);
            L.ffn_down = create_tensor(tn(LLM_TENSOR_MTP_FFN_DOWN, "weight", (int)il), {mtp_n_ff, mtp_n_embd}, 0);

            L.out_scale = create_tensor(tn(LLM_TENSOR_MTP_LAYER_SCALAR, "weight", (int)il), {1u}, TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// get 2D slice view from a 3D tensor, the idx corresponds to the 3rd dim
static ggml_tensor * ggml_view_2d_slice(ggml_context * ctx0, ggml_tensor * x, int idx) {
    GGML_ASSERT(idx < (int) x->ne[2]);
    return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1], ggml_row_size(x->type, x->ne[0]),
                        idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
}

// =========================================================================
// build_cross_attn_no_cache
//
// Scaled-dot-product attention between drafter Q and externally-bound main-
// model K/V, with no KV cache writes and no mask. Mirrors the structure of
// llm_graph_context::build_attn_mha non-flash, no-streams path, stripped of
// soft-cap / alibi / MLA / stream-recombination logic.
//
// Inputs (post-normalization, post-RoPE):
//   q: [head_dim, n_head,    n_tokens]   from drafter
//   k: [head_dim, n_head_kv, kv_len  ]   from main model (cross-attention source)
//   v: [head_dim, n_head_kv, kv_len  ]   from main model
//
// Output (flat, ready for wo):
//   [head_dim * n_head, n_tokens]
//
// GQA: n_head must be a multiple of n_head_kv; ggml_mul_mat handles the
// broadcast across the third dim when this is true.
//
// !! IMPORTANT !! The K/V tensors passed here are expected to be the FULL
// accumulated context from the main model's last sliding/full attention layer,
// not a single ubatch slice. Current capture in graph::graph captures the
// per-ubatch Kcur/Vcur which is only correct when the main model processes
// the full prefill in one ubatch. For true generation (decode one token at a
// time), the driver needs to pull K/V from the main model's KV cache instead.
// =========================================================================
static ggml_tensor * build_cross_attn_no_cache(
        ggml_context  * ctx0,
        ggml_cgraph   * gf,
        ggml_tensor   * q,
        ggml_tensor   * k,
        ggml_tensor   * v,
        ggml_tensor   * mask,   // [kv_max, n_tokens] F32, 0 valid, -INF invalid
        float           kq_scale,
        int             il)
{
    GGML_UNUSED(il);

    // permute to (D, T, H, 1) layout for the matmuls
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);  // [head_dim, kv_len,   n_head_kv]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);  // [head_dim, kv_len,   n_head_kv]

    // kq = K^T · Q → [kv_len, n_tokens, n_head]
    ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

    // softmax with scaling + mask. Mask is [kv_max, n_tokens]; ggml broadcasts
    // it across the head dim. Padded K positions get -INF so they contribute
    // zero probability post-softmax.
    kq = ggml_soft_max_ext(ctx0, kq, mask, kq_scale, /*max_bias=*/0.0f);

    // V needs to be transposed for the second matmul:
    //   V: [head_dim, kv_len, n_head_kv]  →  [kv_len, head_dim, n_head_kv]
    v = ggml_cont(ctx0, ggml_transpose(ctx0, v));

    // attn = V_t · kq → [head_dim, n_tokens, n_head]
    ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);

    // back to (D, H, T) → flatten heads
    ggml_tensor * out = ggml_permute(ctx0, kqv, 0, 2, 1, 3);          // [head_dim, n_head, n_tokens]
    out = ggml_cont_2d(ctx0, out, out->ne[0] * out->ne[1], out->ne[2]); // [head_dim * n_head, n_tokens]

    ggml_build_forward_expand(gf, out);
    return out;
}

llama_model_gemma4::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        n_embd_per_layer(model.hparams.n_embd_per_layer) {
    // For Gemma4 MTP: identify the LAST sliding-attn layer and LAST full-attn
    // layer that compute their own K/V (not kv-shared). The drafter's
    // shared_kv_states pull from these specific layers at runtime.
    int last_kv_swa  = -1;
    int last_kv_full = -1;
    const bool capture_mtp_kv =
        hparams.nextn_predict_layers > 0 && hparams.mtp_n_embd > 0;
    if (capture_mtp_kv) {
        for (int i = 0; i < (int) hparams.n_layer; ++i) {
            if (!hparams.has_kv(i)) continue;
            if (hparams.is_swa(i)) last_kv_swa  = i;
            else                    last_kv_full = i;
        }
    }
    LLAMA_LOG_INFO("gemma4: capture_mtp_kv=%d mtp_n_embd=%u nextn=%u last_kv_swa=%d last_kv_full=%d n_layer=%u\n",
        (int) capture_mtp_kv, hparams.mtp_n_embd, hparams.nextn_predict_layers,
        last_kv_swa, last_kv_full, hparams.n_layer);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    cb(inpL, "inp_scaled", -1);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // TODO: is causal == true correct? might need some changes
    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * inp_per_layer = nullptr;
    if (model.per_layer_tok_embd) {
        inp_per_layer = build_inp_per_layer();
        ggml_build_forward_expand(gf, inp_per_layer);

        // inp_per_layer shape: [n_embd_per_layer, n_tokens, n_layer]
        inp_per_layer = project_per_layer_inputs(inpL, inp_per_layer);
    }

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = nullptr;
        if (!hparams.is_swa(il)) {
            // full_attention layers use rope_freqs for proportional rope
            freq_factors = model.layers[il].rope_freqs;
        }

        // Q projection (shared for both non-KV and KV layers)
        // this is to mirror Gemma4Attention in pytorch code
        ggml_tensor * Qcur;
        {
            Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
            cb(Qcur, "Qcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_pos", il);
        }

        // self-attention
        if (hparams.has_kv(il)) {
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = model.layers[il].wv
                                    ? build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s)
                                    : Kcur; // if v_proj is not present, use Kcur as Vcur
            cb(Vcur, "Vcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);

            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Kcur, "Kcur_pos", il);

            cur = build_attn(inp_attn, model.layers[il].wo,
                    nullptr, model.layers[il].wo_s, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    hparams.f_attention_scale, il);

            // Capture for Gemma4-style MTP drafter (cross-attention K/V).
            // We read VIEW tensors of the FULL accumulated KV cache at this
            // layer (via mctx->get_{swa,base}()->get_k/get_v), which is what
            // the HF reference uses as shared_kv_states. This must be AFTER
            // build_attn() so the cache contains the just-appended Kcur/Vcur.
            //
            // Only emit from the LAST non-kv-shared layer of each attention
            // type — that's where store_full_length_kv is true in HF.
            if (capture_mtp_kv && (il == last_kv_swa || il == last_kv_full)) {
                fprintf(stderr, "[DBG] gemma4 capture-probe il=%d: inp_attn=%p mctx=%p\n",
                    il, (const void*)inp_attn,
                    (const void*)(inp_attn ? inp_attn->mctx : nullptr));
            }
            if (capture_mtp_kv && inp_attn && inp_attn->mctx) {
                fprintf(stderr, "[DBG-A] il=%d entered outer if (capture_mtp_kv && inp_attn && mctx)\n", il);
                // Materialize copies of the cache views via ggml_dup so the
                // backend scheduler tracks them and they appear as real
                // compute nodes (the raw view tensors are not registered).
                if (il == last_kv_swa) {
                    fprintf(stderr, "[DBG-B] il=%d entering swa branch\n", il);
                    const auto * kv_swa = inp_attn->mctx->get_swa();
                    fprintf(stderr, "[DBG-C] il=%d kv_swa=%p\n", il, (const void*)kv_swa);
                    if (kv_swa) {
                        ggml_tensor * raw_K = kv_swa->get_k(ctx0, il);
                        ggml_tensor * raw_V = kv_swa->get_v(ctx0, il);
                        fprintf(stderr, "[DBG-D] il=%d swa: raw_K=%p (ne=[%lld,%lld,%lld]) raw_V=%p\n",
                            il, (const void*)raw_K,
                            raw_K ? (long long)raw_K->ne[0] : 0LL,
                            raw_K ? (long long)raw_K->ne[1] : 0LL,
                            raw_K ? (long long)raw_K->ne[2] : 0LL,
                            (const void*)raw_V);
                        res->t_shared_K_swa = ggml_dup(ctx0, raw_K);
                        res->t_shared_V_swa = ggml_dup(ctx0, raw_V);
                        cb(res->t_shared_K_swa, "mtp_shared_K_swa", il);
                        cb(res->t_shared_V_swa, "mtp_shared_V_swa", il);
                        ggml_build_forward_expand(gf, res->t_shared_K_swa);
                        ggml_build_forward_expand(gf, res->t_shared_V_swa);
                    }
                } else if (il == last_kv_full) {
                    fprintf(stderr, "[DBG-E] il=%d entering full branch\n", il);
                    const auto * kv_base = inp_attn->mctx->get_base();
                    fprintf(stderr, "[DBG-F] il=%d kv_base=%p\n", il, (const void*)kv_base);
                    if (kv_base) {
                        ggml_tensor * raw_K = kv_base->get_k(ctx0, il);
                        ggml_tensor * raw_V = kv_base->get_v(ctx0, il);
                        fprintf(stderr, "[DBG-G] il=%d full: raw_K=%p (ne=[%lld,%lld,%lld]) raw_V=%p\n",
                            il, (const void*)raw_K,
                            raw_K ? (long long)raw_K->ne[0] : 0LL,
                            raw_K ? (long long)raw_K->ne[1] : 0LL,
                            raw_K ? (long long)raw_K->ne[2] : 0LL,
                            (const void*)raw_V);
                        res->t_shared_K_full = ggml_dup(ctx0, raw_K);
                        res->t_shared_V_full = ggml_dup(ctx0, raw_V);
                        cb(res->t_shared_K_full, "mtp_shared_K_full", il);
                        cb(res->t_shared_V_full, "mtp_shared_V_full", il);
                        ggml_build_forward_expand(gf, res->t_shared_K_full);
                        ggml_build_forward_expand(gf, res->t_shared_V_full);
                    }
                }
            }
        } else {
            // reuse KV cache of earlier layers
            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        // TODO @ngxson : strip unused token right after the last KV layer to speed up prompt processing
        // When capturing the post-norm hidden state for the MTP drafter we
        // DEFER this filter until after the final output norm — the drafter
        // needs h at every prompt position to drive each verify step, not
        // just the n_outputs sampled rows. The filter is re-applied below
        // before lm_head so logits cost is unchanged.
        if (il == n_layer - 1 && inp_out_ids && !capture_mtp_kv) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // feed-forward network
        const bool is_moe_layer = model.layers[il].ffn_gate_inp != nullptr;
        if (is_moe_layer) {
            // MLP (shared exp)
            ggml_tensor * cur_mlp = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_norm_1", il);

            cur_mlp = build_ffn(cur_mlp,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cur_mlp = build_norm(cur_mlp,
                    model.layers[il].ffn_post_norm_1, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_mlp", il);

            // Expert FFN
            ggml_tensor * cur_moe = build_norm(attn_out,
                    model.layers[il].ffn_pre_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_norm_2", il);

            // custom MoE logits calculation (router operates on attn_out, not cur)
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));
            tmp = ggml_mul(ctx0, tmp, model.layers[il].ffn_gate_inp_s);
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp); // [n_expert, n_tokens]
            cb(logits, "ffn_moe_logits", il);

            cur_moe = build_moe_ffn(cur_moe,
                    nullptr, // gate_inp
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr, // exp_probs_b (not used for gemma4)
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    1.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il, logits,
                    model.layers[il].ffn_gate_up_exps,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cur_moe = build_norm(cur_moe,
                    model.layers[il].ffn_post_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_moe", il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);
            cb(cur, "ffn_moe_combined", il);
        } else {
            cur = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur,
                model.layers[il].ffn_post_norm, nullptr,
                LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        // residual connection
        cur = ggml_add(ctx0, cur, attn_out);

        // per-layer embedding
        if (inp_per_layer) {
            ggml_tensor * pe_in = cur;
            cb(cur, "pe_in", il);

            cur = build_lora_mm(model.layers[il].per_layer_inp_gate, cur); // [n_embd_per_layer, n_tokens]
            cur = ggml_gelu(ctx0, cur);

            ggml_tensor * inp_this_layer = ggml_view_2d_slice(ctx0, inp_per_layer, il); // [n_embd_per_layer, n_tokens]

            // TODO @ngxson : improve this
            if (il == n_layer - 1 && inp_out_ids) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }

            cur = ggml_mul(ctx0, cur, inp_this_layer);
            cur = build_lora_mm(model.layers[il].per_layer_proj, cur); // [n_embd, n_tokens]
            cur = build_norm(cur, model.layers[il].per_layer_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "per_layer_embd_out", il);

            // residual connection
            cur = ggml_add(ctx0, pe_in, cur);
        }

        // layer_scalar
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, nullptr,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // Capture the base model's post-norm hidden state for the MTP drafter's
    // mtp_h_input. We ggml_dup to ensure the scheduler treats this as a
    // compute node rather than a view (so the backend actually materializes
    // it for tensor_get_async). Only emitted when MTP capture is enabled.
    //
    // The capture happens BEFORE the output-row filter (which we deferred
    // from layer n-1 above) so that t_last_hidden_state holds h for every
    // prompt position, not just sampled outputs. The drafter needs h_t for
    // each verify position; the previous implementation only captured 1 row
    // for typical prompts, leaving batch.embd[1..] uninitialized.
    if (capture_mtp_kv) {
        res->t_last_hidden_state = ggml_dup(ctx0, cur);
        ggml_set_name(res->t_last_hidden_state, "mtp_last_hidden_state");
        ggml_build_forward_expand(gf, res->t_last_hidden_state);

        // Now apply the deferred filter so lm_head only runs on output rows.
        if (inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
    }

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// equivalent to get_per_layer_inputs() in python code
// output shape: [n_embd_per_layer, n_layer, n_tokens]
ggml_tensor * llama_model_gemma4::graph::build_inp_per_layer() {
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    ggml_tensor * inp_per_layer;
    float tok_embd_scale = sqrtf((float) n_embd_per_layer);
    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp_per_layer = ggml_get_rows  (ctx0, model.per_layer_tok_embd, inp->tokens);
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, n_tokens);
        inp_per_layer = ggml_scale     (ctx0, inp_per_layer, tok_embd_scale);
        cb(inp_per_layer, "inp_per_layer_selected", -1);

        res->add_input(std::move(inp));
    } else {
        // Multimodal embedding path: use padding token (ID=0) embedding
        // TODO: verify if this is the correct behavior in transformers implementation
        const int64_t embd_size = model.per_layer_tok_embd->ne[0];  // n_embd_per_layer * n_layer

        // Extract and dequantize padding token embedding (row 0)
        ggml_tensor * padding = ggml_view_1d(ctx0, model.per_layer_tok_embd, embd_size, 0);
        inp_per_layer = ggml_cast (ctx0, padding, GGML_TYPE_F32);
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, tok_embd_scale);

        // Reshape to [n_embd_per_layer, n_layer, 1]
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, 1);
        cb(inp_per_layer, "inp_per_layer_multimodal", -1);
    }
    return inp_per_layer;
}

// equivalent to project_per_layer_inputs() in python code
// this calculates the per-layer inputs, so the final tensor shape will have n_layer as the last dim
// inp_batch     shape: [n_embd, n_tokens]
// inp_per_layer shape: [n_embd_per_layer, n_layer, n_tokens] (from build_inp_per_layer)
// output shape: [n_embd_per_layer, n_tokens, n_layer]
ggml_tensor * llama_model_gemma4::graph::project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer) {
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);

    // note: this matrix multiplication will be performed in the input layer (i.e. on the CPU)
    ggml_tensor * per_layer_proj;
    per_layer_proj = ggml_mul_mat   (ctx0, model.per_layer_model_proj, inp_batch);
    per_layer_proj = ggml_scale     (ctx0, per_layer_proj, per_layer_projection_scale);
    per_layer_proj = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_per_layer, n_layer, n_tokens);

    per_layer_proj = build_norm(per_layer_proj, model.per_layer_proj_norm, nullptr, LLM_NORM_RMS, -1);
    cb(per_layer_proj, "per_layer_proj", -1);

    inp_per_layer = ggml_add  (ctx0, per_layer_proj, inp_per_layer);
    inp_per_layer = ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);
    cb(inp_per_layer, "inp_per_layer", -1);

    // permute to shape: [n_embd_per_layer, n_tokens, n_layer]
    inp_per_layer = ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));
    return inp_per_layer;
}


// =========================================================================
// llama_model_gemma4::graph_mtp — Multi-Token Prediction drafter forward.
//
// The drafter (google/gemma-4-...-it-assistant) is a 4-layer cross-attention
// "head" that consumes:
//   - prev backbone hidden state h_in   (backbone_dim)
//   - next-token id              tokens (looked up via main model's tok_embd)
//   - shared K/V from main model        (NOT yet wired — see TODO below)
//
// Architecture (per modeling_gemma4_assistant.py forward()):
//   x = pre_proj(concat(h_in, tok_embd_main))     # 2*backbone → mtp_n_embd
//   for layer in 4 drafter blocks (3 swa + 1 full):
//       x = layer(x, shared_K[layer_type], shared_V[layer_type])
//   x = mtp.norm(x)
//   logits      = mtp.embed_tokens(x)             # tied lm_head at mtp_n_embd
//   t_h_pre_norm = post_proj(x)                   # mtp_n_embd → backbone, fed
//                                                 #  back as next-step h_in
//
// !!! INITIAL VERSION !!! The shared_kv inputs aren't yet plumbed through the
// llm_graph_result and the main pass. This implementation uses a placeholder
// attention that returns a zero tensor of the correct shape, so the graph
// builds and the model loads end-to-end. Numerical outputs are NOT yet
// meaningful — a separate patch will:
//   1. Add res->t_shared_K_swa / V_swa / K_full / V_full slots.
//   2. Capture them in the main-pass graph from the last sliding/full layer.
//   3. Bind them as inputs here and replace the placeholder with real cross-
//      attention via build_attn_no_cache (to be added in llama-graph.cpp).
// =========================================================================
llama_model_gemma4::graph_mtp::graph_mtp(
        const llama_model & model_, const llm_graph_params & params)
    : llm_graph_context(params), model(model_)
{
    const auto & mtp = model.mtp;
    GGML_ASSERT(mtp.pre_proj      && "missing mtp.pre_proj");
    GGML_ASSERT(mtp.post_proj     && "missing mtp.post_proj");
    GGML_ASSERT(mtp.norm          && "missing mtp.norm");
    GGML_ASSERT(mtp.embed_tokens  && "missing mtp.embed_tokens (tied lm_head)");
    GGML_ASSERT(model.tok_embd    && "missing main-model tok_embd");

    const int64_t n_predict    = (int64_t) hparams.nextn_predict_layers;
    const int64_t backbone_dim = (int64_t) hparams.n_embd;
    // mtp_n_embd was kept in the original sketch for readability but is
    // unused in this scope — the per-block constants (head_dim, q_dim) and
    // tensor shapes derive directly from the relevant `mtp.layers[il]`.
    GGML_ASSERT((int64_t) mtp.layers.size() == n_predict);
    GGML_ASSERT(n_predict > 0);

    // --- inputs ---
    auto inp = std::make_unique<llm_graph_input_embd>(backbone_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    // h_in is the recurrent backbone hidden state from the previous MTP step
    // (or main model's last hidden on the first step). Shape [backbone, n_tokens].
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, backbone_dim, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "mtp_h_input");
    res->t_inp_mtp_h_input = inp->embd;

    ggml_tensor * h_in_t  = inp->embd;
    ggml_tensor * tokens_t = inp->tokens;

    res->add_input(std::move(inp));

    // --- shared K/V from main model's last sliding/full attention layers ---
    //
    // Declared as ggml inputs and named so the driver (or unit tests) can find
    // them by name and write data before forward. Sized to cparams.n_ctx as
    // the upper bound — the driver writes only the actual kv_len rows.
    //
    // !! For correctness, these must be filled with the FULL accumulated K/V
    // from the main model's last non-kv-shared sliding (resp. full) attention
    // layer. The current main-pass capture (see graph::graph above) takes the
    // per-ubatch Kcur/Vcur which is only the latest position. The driver-side
    // path that maintains a growing buffer (or extracts from the main KV
    // cache) is still TODO.
    const int64_t kv_max         = (int64_t) cparams.n_ctx;
    const int64_t mtp_n_kv_swa   = (int64_t) hparams.mtp_n_head_kv;        // 16 (31B SWA) / 1 (E2B)
    const int64_t mtp_n_kv_full  = (int64_t) hparams.mtp_global_n_head_kv; //  4 (31B FULL) / 1 (E2B)
    const int64_t hd_swa         = (int64_t) hparams.mtp_n_embd_head_k;    // 256
    const int64_t hd_full        = (int64_t) hparams.mtp_global_head_dim;  // 512

    // Match the KV cache's storage type (F16) so binding doesn't require
    // per-decode dequant. The cross-attention math inside build_cross_attn_no_cache
    // promotes to F32 via ggml_mul_mat_set_prec(GGML_PREC_F32).
    ggml_tensor * shared_K_swa  = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, hd_swa,  mtp_n_kv_swa,  kv_max);
    ggml_tensor * shared_V_swa  = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, hd_swa,  mtp_n_kv_swa,  kv_max);
    ggml_tensor * shared_K_full = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, hd_full, mtp_n_kv_full, kv_max);
    ggml_tensor * shared_V_full = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, hd_full, mtp_n_kv_full, kv_max);
    ggml_set_input(shared_K_swa);  ggml_set_name(shared_K_swa,  "mtp_shared_K_swa");
    ggml_set_input(shared_V_swa);  ggml_set_name(shared_V_swa,  "mtp_shared_V_swa");
    ggml_set_input(shared_K_full); ggml_set_name(shared_K_full, "mtp_shared_K_full");
    ggml_set_input(shared_V_full); ggml_set_name(shared_V_full, "mtp_shared_V_full");

    // Expose to the result interface so the driver can bind data into them
    // via llama_set_input_tensor() without a graph walk by name.
    res->t_inp_mtp_shared_K_swa  = shared_K_swa;
    res->t_inp_mtp_shared_V_swa  = shared_V_swa;
    res->t_inp_mtp_shared_K_full = shared_K_full;
    res->t_inp_mtp_shared_V_full = shared_V_full;

    // Attention mask for the cross-attention softmax: shape [kv_max, n_tokens]
    // F32. ggml_soft_max_ext requires the mask's query-dim to match kq's, so
    // we can't use a [kv_max, 1] broadcast — the driver MUST rebind the mask
    // before each llama_decode whose n_tokens differs from the previous call.
    ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kv_max, n_tokens);
    ggml_set_input(attn_mask);
    ggml_set_name(attn_mask, "mtp_attn_mask");
    res->t_inp_mtp_attn_mask = attn_mask;

    // Look up next-token embedding from the BACKBONE's tok_embd table (backbone_dim).
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.tok_embd, tokens_t);
    cb(tok_embd, "mtp_tok_embd", -1);

    // --- pre_projection ---
    // concat(h_in, tok_embd) along dim 0 → [2*backbone, n_tokens]
    ggml_tensor * concat = ggml_concat(ctx0, h_in_t, tok_embd, /*dim=*/0);
    cb(concat, "mtp_concat", -1);

    // [mtp_n_embd, n_tokens]
    ggml_tensor * cur = build_lora_mm(mtp.pre_proj, concat);
    cb(cur, "mtp_pre_proj", -1);

    // Debug taps for ref-diff harness. Each tap is ggml_dup'd so the scheduler
    // treats it as a real compute node and the backend materializes it before
    // we call ggml_backend_tensor_get_async. Only enabled for tests; the
    // overhead is small (one extra dup per labeled tensor).
    auto dbg = [&](const char * name, ggml_tensor * t) {
        ggml_tensor * d = ggml_dup(ctx0, t);
        ggml_set_name(d, name);
        ggml_build_forward_expand(gf, d);
        res->t_dbg.emplace_back(name, d);
    };
    dbg("pre_projection", cur);

    // Positions for RoPE on the drafter's Q. The drafter's K/V come from the
    // BASE model's KV cache (post-RoPE at their original positions); to keep
    // relative phase consistent in Q @ K^T, the drafter must RoPE its Q at
    // its own input positions using the same rope_theta as the base.
    ggml_tensor * inp_pos = build_inp_pos();

    // --- 4 drafter blocks (3 sliding + 1 full) ---
    for (int64_t il = 0; il < n_predict; ++il) {
        const auto & L      = mtp.layers[(size_t) il];
        const bool   is_swa = hparams.mtp_is_swa((uint32_t) il);
        const int64_t head_dim = is_swa
                ? (int64_t) hparams.mtp_n_embd_head_k
                : (int64_t) hparams.mtp_global_head_dim;
        const int64_t mtp_n_head = (int64_t) hparams.mtp_n_head;
        const int64_t q_dim      = head_dim * mtp_n_head;

        ggml_tensor * residual = cur;

        // pre-attn norm
        ggml_tensor * x = build_norm(cur, L.attn_norm, nullptr, LLM_NORM_RMS, (int) il);
        cb(x, "mtp_attn_norm", (int) il);
        if (il == 0) dbg("L0.input_layernorm", x);

        // Q projection + q_norm (per-head)
        ggml_tensor * Qcur_pre = build_lora_mm(L.wq, x);
        if (il == 0) dbg("L0.q_proj", Qcur_pre);
        ggml_tensor * Qcur = ggml_reshape_3d(ctx0, Qcur_pre, head_dim, mtp_n_head, n_tokens);
        Qcur = build_norm(Qcur, L.attn_q_norm, nullptr, LLM_NORM_RMS, (int) il);
        cb(Qcur, "mtp_Qcur", (int) il);
        if (il == 0) dbg("L0.q_norm", Qcur);

        // Apply RoPE on Q at the drafter's positions. K already had RoPE applied
        // in the base model's main pass (it's read from the cache), so we just
        // need to RoPE Q here to make the Q@K^T relative-position-correct.
        // For full-attention layers, base uses rope_freqs (proportional rope);
        // for SWA layers, no freq_factors. We use the rope params from the
        // base hparams since the drafter shares text_config with the base.
        const float freq_base_l  = is_swa
                ? hparams.rope_freq_base_train_swa
                : hparams.rope_freq_base_train;
        const float freq_scale_l = 1.0f;
        const int   n_rot_l      = (int) head_dim;
        ggml_tensor * freq_factors = nullptr;
        // For full-attention drafter layers, pull rope_freqs from the base
        // model's first full-attention layer that has them (TENSOR_DUPLICATED
        // means any full layer holds the same tensor).
        if (!is_swa) {
            for (int bil = 0; bil < (int) model.layers.size(); ++bil) {
                if (!hparams.is_swa((uint32_t) bil) && model.layers[bil].rope_freqs) {
                    freq_factors = model.layers[bil].rope_freqs;
                    break;
                }
            }
        }
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type,
                             n_ctx_orig, freq_base_l, freq_scale_l,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "mtp_Qcur_pos", (int) il);
        if (il == 0) dbg("L0.q_rope", Qcur);

        // Cross-attention to main model's shared K/V (no KV cache writes).
        // K/V are selected by layer type (sliding vs full) and bound externally
        // by the driver before forward.
        ggml_tensor * shared_K = is_swa ? shared_K_swa : shared_K_full;
        ggml_tensor * shared_V = is_swa ? shared_V_swa : shared_V_full;
        // Gemma4 uses `self.scaling = 1.0` (see Gemma4TextAttention) — NO
        // 1/sqrt(head_dim) factor. The standard transformer scaling is
        // absorbed elsewhere (q_norm has the head_dim baked into its scale).
        const float kq_scale = 1.0f;
        ggml_tensor * attn_raw = build_cross_attn_no_cache(
                ctx0, gf, Qcur, shared_K, shared_V, attn_mask, kq_scale, (int) il);
        cb(attn_raw, "mtp_attn_raw", (int) il);
        if (il == 0) dbg("L0.attn_out_pre_o_proj", attn_raw);
        // output projection: [head_dim * n_head, n_tokens] → [mtp_n_embd, n_tokens]
        (void) q_dim;
        ggml_tensor * attn_out = build_lora_mm(L.wo, attn_raw);
        cb(attn_out, "mtp_attn_out", (int) il);
        if (il == 0) dbg("L0.o_proj", attn_out);

        // HF order: post_attention_layernorm(attn_out), THEN residual add.
        // (Originally this norm was after the add, which is wrong for Gemma4.)
        attn_out = build_norm(attn_out, L.attn_post_norm, nullptr, LLM_NORM_RMS, (int) il);
        cur = ggml_add(ctx0, residual, attn_out);

        // FFN with dual pre/post norms (Gemma4 quirk)
        ggml_tensor * ffn_residual = cur;
        ggml_tensor * ffn_in       = build_norm(cur, L.ffn_pre_norm, nullptr, LLM_NORM_RMS, (int) il);
        ggml_tensor * ffn_out      = build_ffn(ffn_in,
                L.ffn_up,   nullptr, nullptr,
                L.ffn_gate, nullptr, nullptr,
                L.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_GELU, LLM_FFN_PAR, (int) il);
        ffn_out = build_norm(ffn_out, L.ffn_post_norm, nullptr, LLM_NORM_RMS, (int) il);
        cur = ggml_add(ctx0, ffn_residual, ffn_out);

        // layer_scalar (Gemma4 per-block scalar multiplier)
        if (L.out_scale) {
            cur = ggml_mul(ctx0, cur, L.out_scale);
        }
        cb(cur, "mtp_block_out", (int) il);
        if (il == 0) dbg("L0.out", cur);
    }

    // --- final norm ---
    cur = build_norm(cur, mtp.norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_norm", -1);
    dbg("model.norm", cur);

    // --- two output paths from `cur` (post-norm, mtp_n_embd dim) ---

    // 1. Recurrent state: post_proj → [backbone_dim, n_tokens]
    //    (driver feeds this back as h_in on the next MTP step)
    ggml_tensor * h_next = build_lora_mm(mtp.post_proj, cur);
    cb(h_next, "h_pre_norm", -1);  // slot name kept for driver compatibility
    res->t_h_pre_norm = h_next;
    dbg("post_projection", h_next);

    // 2. Logits. Two paths depending on the drafter variant:
    //    - Plain tied lm_head (31B):   logits = embed_tokens @ cur
    //    - MaskedEmbedder (E2B): sparse centroid-based top-k lookup over
    //      lm_head.weight, with non-selected positions filled with a
    //      mask_value = min(selected_logits) - 1.0. See HF source:
    //      Gemma4AssistantMaskedEmbedder.forward.
    // Plain lm_head is always computed as t_logits (so llama_get_logits and
    // sched_reserve at any n_tokens both work). When use_ordered_embeddings is
    // true, ALSO compute the sparse masked_embedding outputs (selected_logits
    // and selected_indices for the top-K*vpc tokens). The driver reconstructs
    // the full masked-vocab logits host-side via:
    //   logits[v] = min(selected_logits) - 1.0  for v not in selected_indices
    //   logits[v] = selected_logits[i]          for v == selected_indices[i]
    // This avoids the full-vocab scatter intermediate that crashed graph_reserve.
    ggml_tensor * logits = build_lora_mm(mtp.embed_tokens, cur);

    if (hparams.mtp_use_ordered_embeddings && n_tokens == 1) {
        GGML_ASSERT(mtp.masked_emb_centroids      && "masked_emb_centroids not loaded");
        GGML_ASSERT(mtp.masked_emb_token_ordering && "masked_emb_token_ordering not loaded");
        const int64_t num_centroids = (int64_t) hparams.mtp_num_centroids;          // 2048
        const int64_t top_k         = (int64_t) hparams.mtp_centroid_top_k;         // 32
        const int64_t vocab         = (int64_t) model.vocab.n_tokens();             // 262144
        const int64_t vocab_per_c   = vocab / num_centroids;                        // 128

        // Step 1: centroid_logits = centroids @ cur → [num_centroids, n_tokens]
        ggml_tensor * centroid_logits = build_lora_mm(mtp.masked_emb_centroids, cur);
        cb(centroid_logits, "mtp_centroid_logits", -1);
        dbg("centroids", centroid_logits);

        // Step 2: top_k indices along centroid dim. ggml_top_k returns indices
        // (I32) per row; for [num_centroids, n_tokens] it gives [top_k, n_tokens].
        ggml_tensor * top_k_idx = ggml_top_k(ctx0, centroid_logits, (int) top_k);
        cb(top_k_idx, "mtp_top_k_idx", -1);

        // Step 3: gather token clusters. token_ordering is [vocab] F32 (values
        // are int indices encoded as floats). It's an overlay-context leaf,
        // so we ggml_dup it into the compute graph (the scheduler can't
        // allocate the raw overlay tensor on its own — it needs a node it
        // owns). Then view as [vpc, num_centroids] and gather by top_k_idx.
        ggml_tensor * token_ordering_dup = ggml_dup(ctx0, mtp.masked_emb_token_ordering);
        ggml_tensor * token_ordering_2d  = ggml_reshape_2d(ctx0, token_ordering_dup,
                                                            vocab_per_c, num_centroids);
        // Flatten top_k_idx [top_k, n_tokens] → [top_k * n_tokens] so ne[1]
        // matches token_ordering_2d.ne[2]=1 (same centroid mapping for all
        // tokens). After get_rows we reshape back.
        ggml_tensor * top_k_idx_flat = ggml_reshape_1d(ctx0, top_k_idx, top_k * n_tokens);
        ggml_tensor * selected_canonical_f_flat = ggml_get_rows(ctx0, token_ordering_2d, top_k_idx_flat);
        // Shape after: [vpc, top_k*n_tokens, 1, 1]. Reshape to [vpc, top_k, n_tokens, 1].
        ggml_tensor * selected_canonical_f = ggml_reshape_3d(ctx0, selected_canonical_f_flat,
                                                              vocab_per_c, top_k, n_tokens);
        // Cast F32 -> I32 to use as indices into embed_tokens.
        ggml_tensor * selected_canonical_i = ggml_cast(ctx0, selected_canonical_f, GGML_TYPE_I32);
        // Flatten to a 1D index vector [vpc * top_k * n_tokens] for the next
        // get_rows. Embed_tokens has no batch dim, so b must also have ne[1]=1.
        ggml_tensor * sel_flat_i = ggml_reshape_1d(ctx0, selected_canonical_i,
                                                    vocab_per_c * top_k * n_tokens);

        // Step 4: gather embeddings for selected tokens.
        ggml_tensor * selected_embeddings_flat = ggml_get_rows(ctx0, mtp.embed_tokens, sel_flat_i);
        // shape after: [mtp_n_embd, vpc*top_k*n_tokens, 1, 1]. Reshape to
        // [mtp_n_embd, vpc*top_k, n_tokens, 1] so the mul_mat broadcast works.
        ggml_tensor * selected_embeddings = ggml_reshape_3d(ctx0, selected_embeddings_flat,
                                                             (int64_t) hparams.mtp_n_embd,
                                                             vocab_per_c * top_k,
                                                             n_tokens);

        // Step 5: dot product with cur. cur is [mtp_n_embd, n_tokens]; we want
        // selected_logits[i, t] = sum_d cur[d, t] * selected_embeddings[d, i, t]
        // Use ggml_mul_mat with broadcasting: cur needs middle dim of 1.
        ggml_tensor * cur_b = ggml_reshape_3d(ctx0, cur, (int64_t) hparams.mtp_n_embd, 1, n_tokens);
        ggml_tensor * selected_logits_3d = ggml_mul_mat(ctx0, selected_embeddings, cur_b);
        ggml_mul_mat_set_prec(selected_logits_3d, GGML_PREC_F32);
        // shape: [vpc*top_k, 1, n_tokens] — squeeze middle dim
        ggml_tensor * selected_logits = ggml_reshape_2d(ctx0, selected_logits_3d,
                                                         vocab_per_c * top_k, n_tokens);
        cb(selected_logits, "mtp_selected_logits", -1);
        dbg("selected_logits", selected_logits);

        // SPARSE OUTPUTS: expose the two ingredients (selected_logits and the
        // corresponding vocab indices) as debug taps. The harness/driver
        // reconstructs the full masked-vocab logits host-side:
        //   logits[v] = selected_logits[i] if v == sel_flat_i[i] else min(selected_logits) - 1.0
        // This avoids the large-allocation scatter that broke graph_reserve.
        //
        // sel_flat_i is I32; cast to F32 so it flows through the same F32 tap
        // extraction path (the harness reinterprets as int).
        ggml_tensor * sel_indices_f32 = ggml_cast(ctx0, sel_flat_i, GGML_TYPE_F32);
        cb(sel_indices_f32, "mtp_selected_indices", -1);
        dbg("selected_indices", sel_indices_f32);
    }
    cb(logits, "result_output", -1);
    res->t_logits = logits;

    ggml_build_forward_expand(gf, h_next);
    ggml_build_forward_expand(gf, logits);
}
