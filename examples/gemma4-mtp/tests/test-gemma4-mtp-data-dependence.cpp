// tests/test-gemma4-mtp-data-dependence.cpp
//
// Verify the MTP cross-attention is genuinely data-dependent: feed two
// different sets of main-forward tokens, capture two different K/V states,
// run the SAME MTP query, and confirm the argmax differs. If it's the same,
// the cross-attention isn't actually consuming the bound K/V.

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>

static int run_mtp_and_get_argmax(
    llama_model * model,
    const llama_token * main_tokens, int n_main_tokens,
    llama_token draft_token,
    const char * label)
{
    // Fresh contexts each call to avoid state leakage.
    llama_context_params cp_tgt = llama_context_default_params();
    cp_tgt.n_ctx     = 256;
    cp_tgt.n_batch   = 32;
    cp_tgt.n_ubatch  = 32;
    cp_tgt.n_threads = 4;
    llama_context * ctx_tgt = llama_init_from_model(model, cp_tgt);

    llama_context_params cp_dft = llama_context_default_params();
    cp_dft.n_ctx     = 256;
    cp_dft.n_batch   = 32;
    cp_dft.n_ubatch  = 32;
    cp_dft.n_threads = 4;
    cp_dft.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    llama_context * ctx_dft = llama_init_from_model(model, cp_dft);

    // Main forward
    llama_batch b = llama_batch_init(n_main_tokens, 0, 1);
    for (int i = 0; i < n_main_tokens; ++i) {
        b.token[i]    = main_tokens[i];
        b.pos[i]      = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0]= 0;
        b.logits[i]   = (i == n_main_tokens - 1);
    }
    b.n_tokens = n_main_tokens;
    llama_decode(ctx_tgt, b);

    // Bind shared K/V
    const float * K_swa  = llama_get_shared_kv_K_swa (ctx_tgt);
    const float * V_swa  = llama_get_shared_kv_V_swa (ctx_tgt);
    const float * K_full = llama_get_shared_kv_K_full(ctx_tgt);
    const float * V_full = llama_get_shared_kv_V_full(ctx_tgt);
    size_t n_swa  = llama_get_shared_kv_swa_size (ctx_tgt);
    size_t n_full = llama_get_shared_kv_full_size(ctx_tgt);
    llama_set_input_tensor(ctx_dft, "mtp_shared_K_swa",  K_swa,  n_swa  * sizeof(float));
    llama_set_input_tensor(ctx_dft, "mtp_shared_V_swa",  V_swa,  n_swa  * sizeof(float));
    llama_set_input_tensor(ctx_dft, "mtp_shared_K_full", K_full, n_full * sizeof(float));
    llama_set_input_tensor(ctx_dft, "mtp_shared_V_full", V_full, n_full * sizeof(float));

    // MTP decode (same draft token both runs)
    llama_batch db = llama_batch_init(1, 0, 1);
    db.token[0]    = draft_token;
    db.pos[0]      = 0;
    db.n_seq_id[0] = 1;
    db.seq_id[0][0]= 0;
    db.logits[0]   = 1;
    db.n_tokens    = 1;
    llama_decode(ctx_dft, db);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    float * logits = llama_get_logits(ctx_dft);
    int argmax = 0;
    float maxv = logits[0];
    for (uint32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > maxv) { maxv = logits[i]; argmax = i; }
    }
    std::printf("  %s: main_tokens=[", label);
    for (int i = 0; i < n_main_tokens; ++i) std::printf("%s%d", i?",":"", main_tokens[i]);
    std::printf("] draft=%d → argmax=%d (logit=%.3f)\n",
        draft_token, argmax, maxv);

    llama_batch_free(db);
    llama_batch_free(b);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);

    return argmax;
}

int main(int argc, char ** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s BASE OVERLAY\n", argv[0]); return 2; }
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    int rc = llama_model_load_mtp_overlay(model, argv[2]);
    if (rc != 0) { std::fprintf(stderr, "FAIL overlay rc=%d\n", rc); return 1; }

    std::printf("\n== Data-dependence test ==\n");

    // Run A: main tokens [2, 1234, 5678, 9012], draft 100
    const llama_token a[] = { 2, 1234, 5678, 9012 };
    int argmax_a = run_mtp_and_get_argmax(model, a, 4, 100, "A");

    // Run B: same draft 100, but DIFFERENT main tokens
    const llama_token b[] = { 2, 7777, 8888, 9999 };
    int argmax_b = run_mtp_and_get_argmax(model, b, 4, 100, "B");

    // Run C: identical to A (determinism check)
    int argmax_c = run_mtp_and_get_argmax(model, a, 4, 100, "C(=A)");

    std::printf("\n== Verdict ==\n");
    std::printf("  A vs B (different main inputs, same draft): %s\n",
        argmax_a != argmax_b ? "DIFFERENT ✓ (cross-attention is data-dependent)"
                              : "SAME ✗ (cross-attention NOT seeing bound K/V)");
    std::printf("  A vs C (identical inputs, repeat):           %s\n",
        argmax_a == argmax_c ? "SAME ✓ (deterministic)"
                              : "DIFFERENT ✗ (nondeterminism — bug)");

    llama_model_free(model);
    llama_backend_free();
    return (argmax_a != argmax_b && argmax_a == argmax_c) ? 0 : 1;
}
