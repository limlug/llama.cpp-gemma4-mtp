// tests/test-gemma4-mtp-mini-spec.cpp
//
// Mini speculative-decoding driver:
//   1. Load base + MTP overlay
//   2. Tokenize a real prompt
//   3. Run main forward through ctx_tgt
//   4. Sample the next token from ctx_tgt's logits (the "tgt_next" — what the
//      base model would naturally predict)
//   5. Extract shared K/V; bind into ctx_dft
//   6. Run MTP N times to produce N draft tokens
//   7. Print: prompt | tgt_next | draft tokens
//
// This is the smallest test that exercises the SAME flow as
// common_speculative_impl_draft_mtp would in real speculative decoding. The
// quality of the draft tokens (do they look like plausible continuations?)
// is a strong end-to-end correctness signal.

#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

static int sample_argmax(const float * logits, uint32_t n) {
    int best = 0; float bv = logits[0];
    for (uint32_t i = 1; i < n; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

int main(int argc, char ** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s BASE.gguf OVERLAY.gguf\n", argv[0]); return 2; }

    llama_backend_init();

    std::printf("== Load model + overlay ==\n");
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 1; }
    int rc = llama_model_load_mtp_overlay(model, argv[2]);
    if (rc != 0) { std::fprintf(stderr, "overlay failed rc=%d\n", rc); return 1; }
    std::printf("  ok\n");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const uint32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenize a small prompt.
    const char * prompt = "The quick brown fox";
    std::vector<llama_token> tokens(64);
    int n_tokens = llama_tokenize(vocab, prompt, (int) std::strlen(prompt),
                                  tokens.data(), (int32_t) tokens.size(),
                                  /*add_special=*/true, /*parse_special=*/true);
    if (n_tokens <= 0) { std::fprintf(stderr, "tokenize failed (%d)\n", n_tokens); return 1; }
    tokens.resize(n_tokens);

    std::printf("\n== Prompt ==\n  \"%s\"\n  tokens: [", prompt);
    for (int i = 0; i < n_tokens; ++i) std::printf("%s%d", i?",":"", tokens[i]);
    std::printf("]\n");

    // ctx_tgt
    llama_context_params cp_tgt = llama_context_default_params();
    cp_tgt.n_ctx = 256; cp_tgt.n_batch = 32; cp_tgt.n_ubatch = 32; cp_tgt.n_threads = 4;
    llama_context * ctx_tgt = llama_init_from_model(model, cp_tgt);

    // ctx_dft
    llama_context_params cp_dft = llama_context_default_params();
    cp_dft.n_ctx = 256; cp_dft.n_batch = 32; cp_dft.n_ubatch = 32; cp_dft.n_threads = 4;
    cp_dft.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    llama_context * ctx_dft = llama_init_from_model(model, cp_dft);

    // Main forward on the prompt
    std::printf("\n== Main forward (ctx_tgt) ==\n");
    llama_batch b = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        b.token[i] = tokens[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = (i == n_tokens - 1);
    }
    b.n_tokens = n_tokens;
    int dr = llama_decode(ctx_tgt, b);
    std::printf("  rc=%d  (last-pos logits available)\n", dr);
    if (dr != 0) return 1;

    // Target's "natural" next token
    float * tgt_logits = llama_get_logits(ctx_tgt);
    int tgt_next = sample_argmax(tgt_logits, n_vocab);
    char tgt_text[128] = {0};
    int tgt_text_n = llama_token_to_piece(vocab, tgt_next, tgt_text, sizeof(tgt_text), 0, true);
    std::printf("  tgt next-token argmax = %d → \"%.*s\"\n",
        tgt_next, tgt_text_n > 0 ? tgt_text_n : 0, tgt_text);

    // Bind shared K/V into ctx_dft
    std::printf("\n== Bind shared K/V into ctx_dft ==\n");
    const float * K_swa  = llama_get_shared_kv_K_swa (ctx_tgt);
    const float * V_swa  = llama_get_shared_kv_V_swa (ctx_tgt);
    const float * K_full = llama_get_shared_kv_K_full(ctx_tgt);
    const float * V_full = llama_get_shared_kv_V_full(ctx_tgt);
    size_t n_swa  = llama_get_shared_kv_swa_size (ctx_tgt);
    size_t n_full = llama_get_shared_kv_full_size(ctx_tgt);
    std::printf("  K_swa  %zu floats  K_full %zu floats\n", n_swa, n_full);
    llama_set_input_tensor(ctx_dft, "mtp_shared_K_swa",  K_swa,  n_swa  * sizeof(float));
    llama_set_input_tensor(ctx_dft, "mtp_shared_V_swa",  V_swa,  n_swa  * sizeof(float));
    llama_set_input_tensor(ctx_dft, "mtp_shared_K_full", K_full, n_full * sizeof(float));
    llama_set_input_tensor(ctx_dft, "mtp_shared_V_full", V_full, n_full * sizeof(float));

    // Bind base model's final hidden state as the FIRST mtp_h_input. Without
    // this the drafter runs with garbage; see H_INPUT_BUG.md.
    const float * h0 = llama_get_last_hidden_state(ctx_tgt);
    size_t        h0_n = llama_get_last_hidden_state_size(ctx_tgt);
    if (!h0 || h0_n == 0) {
        std::fprintf(stderr, "FATAL: ctx_tgt has no captured last_hidden_state\n");
        return 1;
    }
    std::printf("  h0: %zu floats (last_hidden_state captured)\n", h0_n);
    llama_set_input_tensor(ctx_dft, "mtp_h_input", h0, h0_n * sizeof(float));

    // Bind mtp_attn_mask: 0 for valid K positions [0..n_tokens), -inf for the
    // zero-padded tail [n_tokens..n_ctx). Without this the drafter's softmax
    // gets diluted across all-zero padded K rows.
    const uint32_t n_ctx_dft = (uint32_t) cp_dft.n_ctx;
    std::vector<float> attn_mask((size_t) n_ctx_dft, -std::numeric_limits<float>::infinity());
    for (int i = 0; i < n_tokens; ++i) attn_mask[i] = 0.0f;
    llama_set_input_tensor(ctx_dft, "mtp_attn_mask", attn_mask.data(),
                           attn_mask.size() * sizeof(float));

    // Sample N draft tokens iteratively. After each MTP step rebind h_in to
    // the drafter's own h_pre_norm output so the recurrent state flows.
    //
    // The drafter does SKIP-1 prediction: given h_t + embedding(t+1), predict
    // token at position t+2. So the FIRST input is the base's predicted t+1
    // (tgt_next), not the last prompt token. Each subsequent step uses the
    // previous draft output.
    // Dump activations + inputs from STEP 0 so a Python diff can compare to
    // HF's real_acts.npz / real_inputs.npz. Triggered via env var.
    const char * dump_dir = std::getenv("MINI_SPEC_DUMP_DIR");
    auto write_blob = [&](std::ofstream & bf, std::ofstream & jf,
                          const std::string & name, const float * data, size_t n_floats,
                          size_t & off, bool first) {
        if (!data || n_floats == 0) return;
        size_t nb = n_floats * sizeof(float);
        bf.write((const char *) data, (std::streamsize) nb);
        if (!first) jf << ",\n";
        jf << "  \"" << name << "\": {\"offset\": " << off
           << ", \"n_bytes\": " << nb
           << ", \"shape\": [" << n_floats << "], \"dtype\": \"float32\"}";
        off += nb;
    };

    std::printf("\n== MTP draft (3 tokens) ==\n");
    llama_token cur_draft_input = tgt_next;
    for (int step = 0; step < 3; ++step) {
        llama_batch db = llama_batch_init(1, 0, 1);
        db.token[0] = cur_draft_input;
        db.pos[0] = step;
        db.n_seq_id[0] = 1;
        db.seq_id[0][0] = 0;
        db.logits[0] = 1;
        db.n_tokens = 1;
        int rc_d = llama_decode(ctx_dft, db);
        if (rc_d != 0) { std::fprintf(stderr, "MTP decode step %d failed rc=%d\n", step, rc_d); break; }

        // For E2B (use_ordered_embeddings=true), sample from the sparse
        // masked-embedding outputs: argmax of selected_logits → look up the
        // canonical vocab id via selected_indices. This matches HF reference
        // exactly. For 31B (no masked_embedding), fall back to plain logits.
        const float * sel_logits  = llama_get_dbg_tap_data(ctx_dft, "selected_logits");
        const float * sel_indices = llama_get_dbg_tap_data(ctx_dft, "selected_indices");
        size_t sel_n = llama_get_dbg_tap_size(ctx_dft, "selected_logits");
        float * dft_logits = llama_get_logits(ctx_dft);

        int dft_next;
        if (sel_logits && sel_indices && sel_n > 0) {
            int best = 0; float bv = sel_logits[0];
            for (size_t i = 1; i < sel_n; ++i) if (sel_logits[i] > bv) { bv = sel_logits[i]; best = (int) i; }
            dft_next = (int) sel_indices[best];  // F32-encoded int
        } else {
            dft_next = sample_argmax(dft_logits, n_vocab);
        }
        char buf[128] = {0};
        int bn = llama_token_to_piece(vocab, dft_next, buf, sizeof(buf), 0, true);
        std::printf("  step %d: input=%d → argmax=%d → \"%.*s\"\n",
            step, cur_draft_input, dft_next, bn > 0 ? bn : 0, buf);

        // After step 0 only, dump everything for HF diff.
        if (step == 0 && dump_dir) {
            std::ofstream bf(std::string(dump_dir) + "/acts_cpp_real.bin", std::ios::binary);
            std::ofstream jf(std::string(dump_dir) + "/acts_cpp_real.json");
            jf << "{\n";
            size_t off = 0;
            bool first = true;
            // h_t (base's last hidden state) — input to drafter
            write_blob(bf, jf, "h_t", h0, h0_n, off, first); first = false;
            // Drafter logits (plain lm_head path; not directly comparable to
            // HF's masked-emb logits but useful for argmax sanity)
            write_blob(bf, jf, "logits_plain", dft_logits, n_vocab, off, first);
            // All dbg taps from the drafter graph
            int n_taps = llama_get_dbg_tap_count(ctx_dft);
            for (int i = 0; i < n_taps; ++i) {
                const char * name = llama_get_dbg_tap_name(ctx_dft, i);
                const float * d = llama_get_dbg_tap_data(ctx_dft, name);
                size_t n = llama_get_dbg_tap_size(ctx_dft, name);
                write_blob(bf, jf, name, d, n, off, first);
            }
            jf << "\n}\n";
            std::printf("  [dump] wrote %s/acts_cpp_real.bin (%zu bytes)\n", dump_dir, off);
        }

        cur_draft_input = dft_next;
        llama_batch_free(db);

        // Rebind mtp_h_input for the next step from the drafter's recurrent
        // h_pre_norm output.
        const float * h_next = llama_get_h_pre_norm(ctx_dft);
        size_t        h_next_n = llama_get_h_pre_norm_size(ctx_dft);
        if (h_next && h_next_n > 0) {
            llama_set_input_tensor(ctx_dft, "mtp_h_input", h_next, h_next_n * sizeof(float));
        } else {
            std::fprintf(stderr, "  warn: no h_pre_norm captured from MTP step %d\n", step);
        }
    }

    llama_batch_free(b);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model);
    llama_backend_free();

    std::printf("\n== DONE ==\n");
    return 0;
}
