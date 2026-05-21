// tests/test-gemma4-mtp-full-flow.cpp
//
// Full speculative-decoding-style flow for the Gemma4 MTP pipeline:
//   1. Load base model
//   2. Attach MTP overlay
//   3. Create ctx_tgt (standard context) + ctx_dft (LLAMA_CONTEXT_TYPE_MTP)
//   4. Run llama_decode on ctx_tgt with a 1-token prompt → main forward
//      captures shared K/V into ctx_tgt's host buffers
//   5. Read shared K/V via llama_get_shared_kv_*
//   6. Bind them into ctx_dft via llama_set_input_tensor
//   7. Run llama_decode on ctx_dft → graph_mtp cross-attends to bound K/V
//   8. Print logits — these should now be meaningful (not garbage)
//
// This mirrors what common_speculative_impl_draft_mtp::process() does in the
// real driver, but in a standalone test where we can inspect intermediates.

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s BASE.gguf OVERLAY.gguf\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    std::printf("== Stage 1: load base ==\n");
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "FAIL load\n"); return 1; }
    std::printf("  ok loaded; n_embd=%u n_vocab=%u\n",
        llama_model_n_embd(model),
        llama_vocab_n_tokens(llama_model_get_vocab(model)));

    std::printf("\n== Stage 2: attach MTP overlay ==\n");
    int32_t rc = llama_model_load_mtp_overlay(model, argv[2]);
    if (rc != 0) { std::fprintf(stderr, "FAIL overlay rc=%d\n", rc); llama_model_free(model); return 1; }
    std::printf("  ok overlay attached\n");

    std::printf("\n== Stage 3: create ctx_tgt + ctx_dft ==\n");
    llama_context_params cp_tgt = llama_context_default_params();
    cp_tgt.n_ctx     = 256;
    cp_tgt.n_batch   = 32;
    cp_tgt.n_ubatch  = 32;
    cp_tgt.n_threads = 4;
    cp_tgt.ctx_type  = LLAMA_CONTEXT_TYPE_DEFAULT;
    llama_context * ctx_tgt = llama_init_from_model(model, cp_tgt);
    if (!ctx_tgt) { std::fprintf(stderr, "FAIL ctx_tgt\n"); llama_model_free(model); return 1; }
    std::printf("  ok ctx_tgt\n");

    llama_context_params cp_dft = llama_context_default_params();
    cp_dft.n_ctx     = 256;
    cp_dft.n_batch   = 32;
    cp_dft.n_ubatch  = 32;
    cp_dft.n_threads = 4;
    cp_dft.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    llama_context * ctx_dft = llama_init_from_model(model, cp_dft);
    if (!ctx_dft) { std::fprintf(stderr, "FAIL ctx_dft\n"); llama_free(ctx_tgt); llama_model_free(model); return 1; }
    std::printf("  ok ctx_dft (MTP)\n");

    std::printf("\n== Stage 4: main forward on ctx_tgt ==\n");
    llama_batch batch = llama_batch_init(/*n_tokens=*/4, /*embd=*/0, /*n_seq_max=*/1);
    const llama_token toks[] = { 2, 1234, 5678, 9012 };  // arbitrary tokens
    for (int i = 0; i < 4; ++i) {
        batch.token[i]    = toks[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0]= 0;
        batch.logits[i]   = (i == 3);  // only need logits for last token
    }
    batch.n_tokens = 4;
    int dr_tgt = llama_decode(ctx_tgt, batch);
    std::printf("  llama_decode(ctx_tgt) = %d %s\n", dr_tgt, dr_tgt == 0 ? "(ok)" : "(FAIL)");
    if (dr_tgt != 0) goto cleanup;

    std::printf("\n== Stage 5: read shared K/V from ctx_tgt ==\n");
    {
        const float * K_swa  = llama_get_shared_kv_K_swa (ctx_tgt);
        const float * V_swa  = llama_get_shared_kv_V_swa (ctx_tgt);
        const float * K_full = llama_get_shared_kv_K_full(ctx_tgt);
        const float * V_full = llama_get_shared_kv_V_full(ctx_tgt);
        size_t n_swa  = llama_get_shared_kv_swa_size (ctx_tgt);
        size_t n_full = llama_get_shared_kv_full_size(ctx_tgt);
        std::printf("  K_swa  %p  size=%zu floats (%.2f KiB)\n", (const void*)K_swa,  n_swa,  (double)(n_swa *sizeof(float))/1024);
        std::printf("  V_swa  %p  size=%zu floats\n",            (const void*)V_swa,  n_swa);
        std::printf("  K_full %p  size=%zu floats\n",            (const void*)K_full, n_full);
        std::printf("  V_full %p  size=%zu floats\n",            (const void*)V_full, n_full);
        if (n_swa > 0 && K_swa) {
            std::printf("  K_swa first 8: ");
            for (int i = 0; i < 8; ++i) std::printf("%+.3f ", K_swa[i]);
            std::printf("\n");
        }

        std::printf("\n== Stage 6: bind shared K/V into ctx_dft ==\n");
        if (n_swa > 0 && n_full > 0 && K_swa && V_swa && K_full && V_full) {
            bool b1 = llama_set_input_tensor(ctx_dft, "mtp_shared_K_swa",  K_swa,  n_swa  * sizeof(float));
            bool b2 = llama_set_input_tensor(ctx_dft, "mtp_shared_V_swa",  V_swa,  n_swa  * sizeof(float));
            bool b3 = llama_set_input_tensor(ctx_dft, "mtp_shared_K_full", K_full, n_full * sizeof(float));
            bool b4 = llama_set_input_tensor(ctx_dft, "mtp_shared_V_full", V_full, n_full * sizeof(float));
            std::printf("  bound: K_swa=%d V_swa=%d K_full=%d V_full=%d\n", b1, b2, b3, b4);
        } else {
            std::printf("  WARN: target produced no shared K/V (sizes are zero)\n");
            std::printf("        This likely means the main pass's K/V capture didn't fire —\n");
            std::printf("        check that the base model has both sliding + full attn layers\n");
            std::printf("        and that capture_mtp_kv condition holds.\n");
        }
    }

    std::printf("\n== Stage 7: MTP decode on ctx_dft ==\n");
    {
        llama_batch dft_batch = llama_batch_init(/*n_tokens=*/1, /*embd=*/0, /*n_seq_max=*/1);
        dft_batch.token[0]    = 1234;  // arbitrary
        dft_batch.pos[0]      = 0;
        dft_batch.n_seq_id[0] = 1;
        dft_batch.seq_id[0][0]= 0;
        dft_batch.logits[0]   = 1;
        dft_batch.n_tokens    = 1;
        int dr_dft = llama_decode(ctx_dft, dft_batch);
        std::printf("  llama_decode(ctx_dft) = %d %s\n", dr_dft, dr_dft == 0 ? "(ok)" : "(FAIL)");

        if (dr_dft == 0) {
            const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            float * logits = llama_get_logits(ctx_dft);
            std::printf("  first 8 logits: ");
            for (int i = 0; i < 8; ++i) std::printf("%+.3f ", logits[i]);
            std::printf("\n");
            int argmax = 0;
            float maxv = logits[0];
            for (uint32_t i = 1; i < n_vocab; ++i) {
                if (logits[i] > maxv) { maxv = logits[i]; argmax = i; }
            }
            std::printf("  argmax token = %d (logit = %.3f)\n", argmax, maxv);
        }
        llama_batch_free(dft_batch);
    }

cleanup:
    llama_batch_free(batch);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model);
    llama_backend_free();
    std::printf("\n== DONE ==\n");
    return 0;
}
