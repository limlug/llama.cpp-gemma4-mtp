// tests/test-gemma4-mtp-e2e.cpp
//
// End-to-end test for the Gemma4 MTP pipeline:
//   1. Load a base Gemma4 model.
//   2. Attach an MTP overlay via llama_model_load_mtp_overlay().
//   3. Verify model->mtp.* fields are populated by attempting to create an
//      MTP-type context (which builds graph_mtp; its GGML_ASSERTs would fire
//      if any required tensor pointer is null).
//
// Usage:
//   test-gemma4-mtp-e2e BASE.gguf OVERLAY.gguf
//
// Exit code: 0 on success, non-zero on any failure.

#include "llama.h"
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s BASE.gguf OVERLAY.gguf\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    std::printf("== Stage 1: load base ==\n");
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;  // CPU only for predictable behavior
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "FAIL: could not load base from %s\n", argv[1]);
        return 1;
    }
    std::printf("  ok  loaded\n");

    std::printf("\n== Stage 2: attach MTP overlay ==\n");
    int32_t rc = llama_model_load_mtp_overlay(model, argv[2]);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: llama_model_load_mtp_overlay returned %d\n", rc);
        llama_model_free(model);
        return 1;
    }
    std::printf("  ok  overlay attached (rc=0)\n");

    std::printf("\n== Stage 3: build MTP context ==\n");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx       = 256;
    cp.n_batch     = 32;
    cp.n_ubatch    = 32;
    cp.n_threads   = 4;
    cp.ctx_type    = LLAMA_CONTEXT_TYPE_MTP;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: could not create MTP context\n");
        llama_model_free(model);
        return 1;
    }
    std::printf("  ok  MTP context created\n");

    std::printf("\n== Stage 4: smoke test bind API ==\n");
    // Just bind some dummy data to confirm the binding path doesn't crash.
    static float dummy[1024];
    for (size_t i = 0; i < 1024; ++i) dummy[i] = 0.0f;
    bool b = llama_set_input_tensor(ctx, "mtp_shared_K_swa", dummy, sizeof(dummy));
    std::printf("  bind 'mtp_shared_K_swa' (small dummy): %s\n", b ? "ok" : "fail");
    llama_clear_input_tensor_bindings(ctx);
    std::printf("  cleared bindings ok\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    std::printf("\n== ALL STAGES PASSED ==\n");
    return 0;
}
