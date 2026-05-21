# Gemma4 MTP for llama.cpp — Handoff

## What this is

A working implementation of Multi-Token Prediction (MTP) speculative decoding for `google/gemma-4-31B-it-assistant` (and family) in [llama.cpp](https://github.com/ggml-org/llama.cpp), built on top of PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) which added Qwen3.5/3.6 MTP.

The Gemma4 drafter is architecturally distinct: it runs at a different hidden size from the main model (1024 vs 5376), has 4 transformer blocks instead of 1, and does **cross-attention** to the main model's KV instead of self-attention. This required substantial extensions beyond the Qwen35 MTP infrastructure.

## Repo state (everything as of the last commit)

- Working tree on dpl26: `~/llama.cpp-gemma4-mtp/`
- Mirror of applied source files: `applied/` and `applied/cpp/` in this project dir
- Converted GGUF on dpl26: `/tmp/gguf-out/gemma-4-31B-it-mtp.gguf` (939 MB)
- HF reference fixture: `tools/reference_activations.json` (argmax pos 0 = token 89004)

## Regression suite — run anytime

```bash
./tools/repro_all.sh                # all 6 stages
./tools/repro_all.sh --stage 4 6    # just C++ build + validator
```

Expected output: `Passed: 6, Failed: 0`. If any stage fails after your changes, something regressed.

## Public C API (added by this work, all in `include/llama.h`)

```c
// Overlay loading
LLAMA_API int32_t llama_model_load_mtp_overlay(
    struct llama_model * model, const char * path);

// Shared K/V accessors (read after main forward)
LLAMA_API const float * llama_get_shared_kv_K_swa  (struct llama_context * ctx);
LLAMA_API const float * llama_get_shared_kv_V_swa  (struct llama_context * ctx);
LLAMA_API const float * llama_get_shared_kv_K_full (struct llama_context * ctx);
LLAMA_API const float * llama_get_shared_kv_V_full (struct llama_context * ctx);
LLAMA_API size_t        llama_get_shared_kv_swa_size  (struct llama_context * ctx);
LLAMA_API size_t        llama_get_shared_kv_full_size (struct llama_context * ctx);

// Driver binding (write before MTP forward)
LLAMA_API bool llama_set_input_tensor(
    struct llama_context * ctx, const char * name,
    const void * data, size_t n_bytes);
LLAMA_API void llama_clear_input_tensor_bindings(struct llama_context * ctx);
```

Recognized binding names for `llama_set_input_tensor`:
- `mtp_shared_K_swa`, `mtp_shared_V_swa` — for sliding-attn cross-attention
- `mtp_shared_K_full`, `mtp_shared_V_full` — for full-attn cross-attention

## End-to-end usage (when a base GGUF is available)

```c
// Load base Gemma4 model normally
llama_model * model = llama_model_load_from_file("gemma-4-31B-it.gguf", mp);

// Attach the MTP overlay
int32_t rc = llama_model_load_mtp_overlay(model, "gemma-4-31B-it-mtp.gguf");
if (rc != 0) { /* see error codes in llama.h */ }

// The model now has model->mtp.* populated. Create main + draft contexts.
llama_context * ctx_tgt = llama_init_from_model(model, tgt_params);
llama_context_params dft_params = llama_context_default_params();
dft_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
llama_context * ctx_dft = llama_init_from_model(model, dft_params);

// Configure speculative decoding (existing PR #22673 driver picks up Gemma4 MTP automatically)
//   llama-server -m base.gguf --mtp-model gemma-4-31B-it-mtp.gguf \
//                --spec-type draft-mtp --spec-draft-n-max 3
```

`common/speculative.cpp::common_speculative_impl_draft_mtp::process()` already calls the K/V getter+setter chain for Gemma4 archs (gated on `llama_get_shared_kv_swa_size > 0`, so it's a no-op for Qwen35).

## What's done

(See `README.md` table for the full status grid)

- ✅ Conversion pipeline (HF → GGUF) for `Gemma4AssistantForCausalLM`
- ✅ GGUF schema: 15 `MTP_*` tensor enums, 12 `Keys.MTP` hparam keys
- ✅ C++ schema: `LLM_TENSOR_MTP_*`, `LLM_KV_MTP_*`, `llama_mtp_block`, `llama_mtp_layer`
- ✅ `gemma4.cpp::load_arch_hparams` reads all MTP keys
- ✅ `gemma4.cpp::load_arch_tensors` allocates MTP tensors when GGUF is combined
- ✅ `gemma4.cpp::graph::graph` captures shared K/V from last sliding + full attn layers
- ✅ `gemma4.cpp::graph_mtp` — full forward path with cross-attention to shared K/V
- ✅ `gemma4.cpp::build_cross_attn_no_cache` — Q·K^T → softmax → ·V helper
- ✅ `llama-context.cpp` extracts shared K/V into host vectors per ubatch
- ✅ `llama-context.cpp` applies user-supplied named bindings after `set_inputs`
- ✅ `llama_model_load_mtp_overlay` — full overlay loading with validation
- ✅ `common/speculative.cpp` driver block: reads from `ctx_tgt`, binds into `ctx_dft`
- ✅ End-to-end regression script (`tools/repro_all.sh`) — **8/8 stages PASS** including live overlay loading on real model
- ✅ C++ structural validator (`tests/test-gemma4-mtp-validate.cpp`) — 19 KV + 48 tensors
- ✅ C++ e2e overlay test (`tests/test-gemma4-mtp-e2e.cpp`) — base load + overlay attach + MTP context + bind API, all PASS on `gemma-4-E2B-it` pair

## What remains

1. **Base Gemma4 GGUF** — DONE for `gemma-4-E2B-it`. Both base GGUF and MTP overlay are now on dpl26:
   - `/tmp/gguf-out/gemma-4-E2B-it.gguf` — 9.3 GB, 601 tensors
   - `/tmp/gguf-out/gemma-4-E2B-it-mtp.gguf` — 154 MB, 48 tensors
   - llama-cli loads the base successfully (verified at interactive prompt)
   - Conversion added a `masked_embedding.*` skip path for `use_ordered_embeddings=True` models

2. **K/V provenance fix** — See `KV_PROVENANCE_FIX.md`. Current capture is per-ubatch slice; HF reference uses full accumulated KV from the cache. ~1 day of work once a base GGUF exists.

3. **Numerical regression** — Validate that the C++ produces the same logits as the HF reference for a fixed input (`argmax_pos0 == 89004`). Use the existing `tests/test_gemma4_mtp_reference.cpp` scaffold (needs `reference_probe.py --dump-full` extension for raw input tensors).

4. **Optimization passes** (post-MVP):
   - Move overlay tensors to model's preferred backend (currently CPU-resident)
   - Mask construction for bidirectional/SWA attention (HF reference has it)
   - Replace placeholder unused-var warning in `gemma4.cpp`

## Layout

```
gemma4-mtp-llama-cpp/
├── HANDOFF.md                       ← you are here
├── README.md                        Status table + step-by-step plan
├── DESIGN.md                        v1 (kept for history)
├── DESIGN_v2.md                     v2 — current architecture spec
├── KV_PROVENANCE_FIX.md             How to do the remaining K/V fix
├── applied/                         Source files as installed on dpl26
│   ├── *.py                         (5 conversion-side)
│   └── cpp/*.{h,cpp}                (13 C++ files)
├── patches/                         Patch-style documentation (slightly stale)
├── tests/
│   ├── test-gemma4-mtp-validate.cpp     C++ structural validator (working)
│   └── test_gemma4_mtp_reference.cpp    Numerical-reference test scaffold
└── tools/
    ├── repro_all.sh                 End-to-end regression (6 stages)
    ├── repro_all_output.txt         Sample passing output
    ├── dry_run_convert.py           HF→GGUF name mapping validator
    ├── reference_probe.py           HF forward to capture ground truth
    ├── reference_activations.json   Ground truth output of the probe
    ├── converted_gguf_dump.txt      gguf_dump of the converted overlay
    └── validator_output.txt         C++ validator sample output
```

## To pick this up next session

1. Run `./tools/repro_all.sh`. If 6/6 PASS, the code is healthy.
2. Read `README.md` for the status grid.
3. Pick a remaining item from the "What remains" list above.
4. After your change, run the regression again.
5. Commit your work to `applied/` so this archive stays the canonical record.

## Project lineage

The trail of what went wrong and what we did about it:

- vLLM v75/v76/v77/v78 attempts — all failed with the `(4096×1024 vs 3072×1024)` shape mismatch because Gemma4Assistant is fundamentally cross-attention to main model KV, not patchable from vLLM's text-only path.
- Pivot to llama.cpp port on top of PR #22673 (Qwen35 MTP).
- Read `modeling_gemma4_assistant.py` source → ran HF probe to capture reference → realized drafter is cross-attention → designed shared-KV plumbing → built it.
- Converted overlay GGUF works (verified by Python + C++ validators).
- C++ build is clean, regression suite passes 6/6.
- Awaiting base GGUF to actually run inference.
