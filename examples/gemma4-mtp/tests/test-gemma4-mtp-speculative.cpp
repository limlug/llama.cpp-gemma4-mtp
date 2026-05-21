// tests/test-gemma4-mtp-speculative.cpp
//
// Production-style speculative-decoding demo for Gemma4 MTP. Shows the full
// loop: base predicts, drafter speculates k tokens, base verifies the
// sequence and decides accept/reject for each draft. Greedy sampling.
//
// This is a self-contained demonstration — production usage goes through
// common/speculative.cpp via llama-cli or llama-server. The point here is to
// prove the API surface (llama_get_logits returning HF-equivalent masked
// logits, the K/V + h + attn_mask bindings, h_pre_norm recurrent feedback)
// composes into working speculative decoding end-to-end.

#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static int argmax(const float * v, uint32_t n) {
    int b = 0; float bv = v[0];
    for (uint32_t i = 1; i < n; ++i) if (v[i] > bv) { bv = v[i]; b = (int) i; }
    return b;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s BASE.gguf OVERLAY.gguf\n", argv[0]);
        return 2;
    }
    const int N_DRAFT = 3;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    if (llama_model_load_mtp_overlay(model, argv[2]) != 0) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const uint32_t n_vocab = llama_vocab_n_tokens(vocab);

    const char * prompt = "The quick brown fox";
    std::vector<llama_token> tokens(64);
    int n_prompt = llama_tokenize(vocab, prompt, (int) std::strlen(prompt),
                                  tokens.data(), (int32_t) tokens.size(), true, true);
    tokens.resize(n_prompt);

    std::printf("== Speculative decoding demo ==\n");
    std::printf("Prompt: %s\n", prompt);

    llama_context_params cp_tgt = llama_context_default_params();
    cp_tgt.n_ctx = 256; cp_tgt.n_batch = 32; cp_tgt.n_ubatch = 32; cp_tgt.n_threads = 8;
    llama_context * ctx_tgt = llama_init_from_model(model, cp_tgt);

    llama_context_params cp_dft = llama_context_default_params();
    cp_dft.n_ctx = 256; cp_dft.n_batch = 32; cp_dft.n_ubatch = 32; cp_dft.n_threads = 8;
    cp_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    llama_context * ctx_dft = llama_init_from_model(model, cp_dft);

    // 1) Base decodes prompt. logits[-1] predicts next.
    {
        llama_batch b = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            b.token[i]    = tokens[i];
            b.pos[i]      = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0]= 0;
            b.logits[i]   = (i == n_prompt - 1);
        }
        b.n_tokens = n_prompt;
        if (llama_decode(ctx_tgt, b) != 0) return 1;
        llama_batch_free(b);
    }
    int tgt_next = argmax(llama_get_logits(ctx_tgt), n_vocab);
    char buf[256] = {0};
    int bn = llama_token_to_piece(vocab, tgt_next, buf, sizeof(buf), 0, true);
    std::printf("Base predicts t+1: %d → %.*s\n", tgt_next, bn>0?bn:0, buf);

    // 2) Drafter forward — bind K/V + h_t + attn_mask, draft N_DRAFT tokens
    // Verbatim copy of mini-spec's binding block:
    const float * K_swa  = llama_get_shared_kv_K_swa(ctx_tgt);
    const float * V_swa  = llama_get_shared_kv_V_swa(ctx_tgt);
    const float * K_full = llama_get_shared_kv_K_full(ctx_tgt);
    const float * V_full = llama_get_shared_kv_V_full(ctx_tgt);
    size_t n_swa  = llama_get_shared_kv_swa_size(ctx_tgt);
    size_t n_full = llama_get_shared_kv_full_size(ctx_tgt);
    std::printf("  K_swa  %zu floats  K_full %zu floats\n", n_swa, n_full);
    llama_set_input_tensor(ctx_dft, "mtp_shared_K_swa",  K_swa,  n_swa  * sizeof(float));
    llama_set_input_tensor(ctx_dft, "mtp_shared_V_swa",  V_swa,  n_swa  * sizeof(float));
    llama_set_input_tensor(ctx_dft, "mtp_shared_K_full", K_full, n_full * sizeof(float));
    llama_set_input_tensor(ctx_dft, "mtp_shared_V_full", V_full, n_full * sizeof(float));
    const float * h0 = llama_get_last_hidden_state(ctx_tgt);
    size_t h0_n = llama_get_last_hidden_state_size(ctx_tgt);
    if (!h0 || h0_n == 0) {
        std::fprintf(stderr, "FATAL: ctx_tgt has no captured last_hidden_state\n");
        return 1;
    }
    llama_set_input_tensor(ctx_dft, "mtp_h_input", h0, h0_n * sizeof(float));
    const uint32_t n_ctx_dft = (uint32_t) cp_dft.n_ctx;
    std::vector<float> attn_mask((size_t) n_ctx_dft, -std::numeric_limits<float>::infinity());
    for (int i = 0; i < n_prompt; ++i) attn_mask[i] = 0.0f;
    llama_set_input_tensor(ctx_dft, "mtp_attn_mask", attn_mask.data(),
                           attn_mask.size() * sizeof(float));

    std::vector<llama_token> drafts;
    llama_token draft_input = tgt_next;
    for (int d = 0; d < N_DRAFT; ++d) {
        llama_batch db = llama_batch_init(1, 0, 1);
        db.token[0] = draft_input; db.pos[0] = d;
        db.n_seq_id[0] = 1; db.seq_id[0][0] = 0; db.logits[0] = 1; db.n_tokens = 1;
        if (llama_decode(ctx_dft, db) != 0) { llama_batch_free(db); break; }
        llama_batch_free(db);
        int dft_next = argmax(llama_get_logits(ctx_dft), n_vocab);
        // Cross-check: also sample from sparse outputs
        const float * sl = llama_get_dbg_tap_data(ctx_dft, "selected_logits");
        const float * si = llama_get_dbg_tap_data(ctx_dft, "selected_indices");
        size_t sl_n = llama_get_dbg_tap_size(ctx_dft, "selected_logits");
        int dft_sparse = -1;
        if (sl && si && sl_n > 0) {
            int best = 0; float bv = sl[0];
            for (size_t j = 1; j < sl_n; ++j) if (sl[j] > bv) { bv = sl[j]; best = (int) j; }
            dft_sparse = (int) si[best];
        }
        // Sanity: dense (via llama_get_logits's masked reconstruction) and
        // sparse (direct dbg-tap argmax) must agree.
        if (dft_next != dft_sparse) {
            std::fprintf(stderr, "FATAL: dense=%d != sparse=%d\n", dft_next, dft_sparse);
            return 1;
        }
        (void) sl_n;
        drafts.push_back(dft_next);
        // Recurrent h feedback
        const float * h_pre = llama_get_h_pre_norm(ctx_dft);
        if (h_pre) {
            llama_set_input_tensor(ctx_dft, "mtp_h_input",
                h_pre, llama_get_h_pre_norm_size(ctx_dft) * sizeof(float));
        }
        draft_input = dft_next;
    }

    std::printf("\nDrafter speculates %d tokens:\n", (int) drafts.size());
    for (size_t i = 0; i < drafts.size(); ++i) {
        char b2[256] = {0};
        int bn2 = llama_token_to_piece(vocab, drafts[i], b2, sizeof(b2), 0, true);
        std::printf("  draft[%zu] = %d → %.*s\n", i, drafts[i], bn2>0?bn2:0, b2);
    }

    // 3) Base verifies by decoding the speculated sequence [tgt_next, drafts...]
    //    starting at position n_prompt. logits[i] tells what base predicts for
    //    position n_prompt+i+1 given the sequence up through position n_prompt+i.
    std::vector<llama_token> verify_seq = { tgt_next };
    for (auto t : drafts) verify_seq.push_back(t);

    {
        llama_batch b = llama_batch_init((int) verify_seq.size(), 0, 1);
        for (int i = 0; i < (int) verify_seq.size(); ++i) {
            b.token[i]    = verify_seq[i];
            b.pos[i]      = n_prompt + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0]= 0;
            b.logits[i]   = 1;  // need every position for verification
        }
        b.n_tokens = (int) verify_seq.size();
        if (llama_decode(ctx_tgt, b) != 0) return 1;
        llama_batch_free(b);
    }

    // 4) Walk through verification: at index i, base saw tgt_next + drafts[0..i-1]
    //    in its context, and its logits at index i predicted position n_prompt+i+1
    //    (= the token AFTER drafts[i-1], or AFTER tgt_next if i==0).
    //    So drafts[i] is accepted iff argmax(verify_logits_ith[i]) == drafts[i].
    std::vector<llama_token> accepted = { tgt_next };
    int n_accepted_drafts = 0;
    int corrected_token = -1;
    for (size_t i = 0; i < drafts.size(); ++i) {
        float * vlog = llama_get_logits_ith(ctx_tgt, (int) i);
        int v_argmax = argmax(vlog, n_vocab);
        if (v_argmax == drafts[i]) {
            accepted.push_back(drafts[i]);
            n_accepted_drafts++;
        } else {
            corrected_token = v_argmax;
            break;
        }
    }
    // Either all drafts accepted (sample one more from logits[last])
    // or first mismatch at draft i → take base's correction at that position.
    if (corrected_token >= 0) {
        accepted.push_back(corrected_token);
    } else {
        // All drafts accepted — sample the FOLLOWING token from the last verify logits.
        float * vlog = llama_get_logits_ith(ctx_tgt, (int) drafts.size());
        accepted.push_back(argmax(vlog, n_vocab));
    }

    std::printf("\nBase verification:\n");
    std::printf("  drafts accepted: %d / %d\n", n_accepted_drafts, (int) drafts.size());
    if (corrected_token >= 0) {
        char b3[256] = {0};
        int bn3 = llama_token_to_piece(vocab, corrected_token, b3, sizeof(b3), 0, true);
        std::printf("  first mismatch at draft %d → corrected to %d (%.*s)\n",
                    n_accepted_drafts, corrected_token, bn3>0?bn3:0, b3);
    }

    std::printf("\n== Generated sequence ==\n%s", prompt);
    for (auto t : accepted) {
        char b4[256] = {0};
        int bn4 = llama_token_to_piece(vocab, t, b4, sizeof(b4), 0, true);
        std::printf("%.*s", bn4>0?bn4:0, b4);
    }
    std::printf("\n");
    std::printf("Tokens produced: 1 (base) + %d (accepted drafts) + 1 (correction or next) = %zu\n",
                n_accepted_drafts, accepted.size());
    std::printf("Speculative speedup factor for this round: %.2fx\n",
                (double) accepted.size() / 1.0);

    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
