# Known runtime bug: shared K/V extraction from KV cache views

## Symptom

Discovered while running `tests/test-gemma4-mtp-full-flow.cpp`:

```
== Stage 4: main forward on ctx_tgt ==
  llama_decode(ctx_tgt) = 0 (ok)               ← decode succeeds

== Stage 5: read shared K/V from ctx_tgt ==
  K_swa  (nil)  size=0 floats                  ← but K/V buffers are empty!
  V_swa  (nil)  size=0 floats
  K_full (nil)  size=0 floats
  V_full (nil)  size=0 floats
```

The main forward runs cleanly, but the K/V capture path leaves the host-side buffers (`shared_kv_*_swa`, `shared_kv_*_full` in `llama_context`) empty.

## Suspected cause

In `gemma4.cpp::graph::graph`, the K/V provenance fix replaced per-ubatch `Kcur` capture with cache-view tensors:

```cpp
res->t_shared_K_swa = kv_swa->get_k(ctx0, il);  // returns a VIEW tensor
```

In `llama-context.cpp`, the extraction does:

```cpp
auto extract_shared = [&](ggml_tensor * t, std::vector<float> & buf) {
    if (!t) return;
    ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched.get(), t);
    if (!be) return;  // ← suspected silent-skip
    ...
    ggml_backend_tensor_get_async(be, t, buf.data(), 0, n_bytes);
};
```

The hypothesis: **view tensors aren't registered with the backend scheduler**, so `ggml_backend_sched_get_tensor_backend` returns null and we silently skip. The view itself is just a metadata reinterpretation of the cache's underlying buffer; the scheduler doesn't track it separately.

## Fixes to try

### Option A — copy from the underlying cache buffer directly

Instead of going through the scheduler:

```cpp
// View tensor t points into the cache's backend buffer. Read the buffer
// directly via ggml_backend_buffer_get/ggml_backend_buffer_get_tensor.
ggml_backend_buffer_t buf_t = t->buffer;
if (buf_t) {
    ggml_backend_tensor_get(t, buf.data(), 0, n_bytes);  // sync API, may copy
}
```

Risk: blocking; need to ensure the main forward has completed first (which it has, since we run this in post-processing).

### Option B — write a node that *copies* the view to a new tensor

In `graph::graph`, instead of assigning the raw view, materialize a copy:

```cpp
ggml_tensor * shared_K = ggml_dup(ctx0, kv_swa->get_k(ctx0, il));
ggml_set_name(shared_K, "mtp_shared_K_swa_capture");
res->t_shared_K_swa = shared_K;
```

The `dup` produces a fresh tensor in the compute graph, which the scheduler tracks. Trade-off: an extra copy per ubatch.

### Option C — fall back to per-ubatch slice

Revert the K/V provenance fix and accept the per-ubatch slice limitation. Correct for prefill-in-one-ubatch; wrong for autoregressive decode. Documented in `KV_PROVENANCE_FIX.md`.

## Recommended next step

Try **Option B** first — it's the most ggml-idiomatic and 1-line change. If `ggml_dup` doesn't trigger the right scheduler registration, try Option A. Validate by re-running `test-gemma4-mtp-full-flow.cpp` and checking that `K_swa` is non-null with `size > 0`.

## Update: Option B (ggml_dup) tested, didn't fix

Wrapped the captures in `ggml_dup` + `ggml_build_forward_expand`. Rebuilt, re-ran `test-gemma4-mtp-full-flow.cpp`. Result: K/V buffers still empty after main forward.

This indicates the issue is **upstream of the extraction** — the capture path itself isn't running. Hypothesis ranking (most likely first):

1. **`inp_attn->mctx` is null at graph-build time** — `build_attn_inp_kv_iswa()` may return an input with no mctx populated during the graph-reserve phase (before any real ubatch flows through). Our `if (... && inp_attn->mctx)` guards against this and silently no-ops.
2. **`last_kv_swa` / `last_kv_full` stayed at -1** — could happen if `hparams.has_kv(i)` is false for all layers (e.g. `n_layer_kv_from_start` is 0 because `num_kv_shared_layers == n_layer`).
3. **The reserved graph (from `sched_reserve`) is reused** — graph nodes built during reserve don't include our capture; on actual decode, the cached graph is replayed without rebuilding.
4. **`hparams.mtp_n_embd` is 0 at the time `capture_mtp_kv` is computed** — maybe the overlay-loading hparam writes don't propagate to the model's effective hparams used during graph construction.

### Debug plan for next session

The probes from this session showed:
- `capture_mtp_kv = 1` ✓
- `last_kv_swa = 13`, `last_kv_full = 14` ✓
- At il=13 and il=14: `inp_attn != null`, `inp_attn->mctx != null` ✓ (capture-probe before the if-block prints these)
- BUT: the inner `if (capture_mtp_kv && inp_attn && inp_attn->mctx) { if (il == last_kv_swa) { ... } }` block's first `LLAMA_LOG_INFO` does **NOT** print

This is genuinely puzzling — same condition values, same macro, same iteration. Hypotheses for next session:

1. **The capture-probe and the inner if see DIFFERENT iteration scopes.** Maybe graph build is doing a "dry build" first (just allocating ops) and the conditional path is shortcircuited. Check by adding `fprintf(stderr, ...)` instead of `LLAMA_LOG_INFO` to bypass any log filtering.

2. **`inp_attn->mctx` becomes null between the two checks.** Unlikely (no thread; not a writeback target) but cheap to rule out by re-loading the value.

3. **An exception is thrown silently inside `inp_attn->mctx->get_swa()`.** Add try/catch.

4. **The macro expansion is being preprocessor-elided** under some build flag. Print the macro definition site and the expanded text via `g++ -E`.

Concrete next experiment:
```cpp
if (capture_mtp_kv && inp_attn && inp_attn->mctx) {
    fprintf(stderr, "[DBG-A] il=%d entering outer if\n", il);
    if (il == last_kv_swa) {
        fprintf(stderr, "[DBG-B] il=%d entering swa branch\n", il);
        const auto * kv_swa = inp_attn->mctx->get_swa();
        fprintf(stderr, "[DBG-C] il=%d kv_swa=%p\n", il, (void*)kv_swa);
        ...
    }
}
```

Whichever DBG line stops printing tells us where execution diverges.

## Resolution (2026-05-20)

The fprintf probes ran and revealed TWO bugs:

### Bug 1: extract_shared lambda only in encode() path

The `extract_shared` lambda was added to `llama_context::encode()` (line ~1500) but the decode path (`llama_context::decode`, line ~1965+) has its own embd_pre_norm extraction block — that's where the main decode flow runs.

**Fix**: Added `extract_shared_d` lambda to the decode path too, mirroring the encode-side extraction. Now both paths extract.

### Bug 2: F32 input declaration vs F16 cache type

`graph_mtp` declared the four `mtp_shared_K/V_{swa,full}` input tensors as `GGML_TYPE_F32`. But the KV cache stores K/V in `GGML_TYPE_F16` (per the model loader's default). The size mismatch:

```
process_ubatch: size mismatch for input tensor 'mtp_shared_K_full': bound 262144 B, expected 524288 B
```

Bound size (F16 from cache): 262144 = 256·1·256·2.
Expected size (F32 declared): 524288 = 256·1·256·4.

**Fix**: Changed input declarations to `GGML_TYPE_F16`. The cross-attention math (`build_cross_attn_no_cache`) already promotes to F32 via `ggml_mul_mat_set_prec(kq, GGML_PREC_F32)`, so accuracy is unaffected.

## Validation

After both fixes:
- `extract_shared_d(K_swa): t=0x..., n_bytes=131072` ✓
- K_swa host buffer 32768 floats ✓
- `bound: K_swa=1 V_swa=1 K_full=1 V_full=1` ✓ no errors
- MTP decode `argmax token` changed from 236775 (uninit) to 122028 (real K/V) — data-dependent
- 8/8 regression PASS

## What this doesn't change

The rest of the pipeline (graph_mtp, cross_attn_no_cache, binding API, driver wiring) is unaffected. This is a single integration point. Once the fix lands, the existing test exercises the full flow.

The MTP decode in stage 7 still ran and produced logits (just from uninitialized K/V) — proving the graph_mtp construction and execution path is correct.
