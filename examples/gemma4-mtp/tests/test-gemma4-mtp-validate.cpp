// tests/test-gemma4-mtp-validate.cpp
//
// Structural validator for a Gemma4 MTP overlay GGUF.
//
// Opens the file, reads metadata via the gguf C API, and asserts that every
// expected MTP-related KV pair and tensor is present with the right type and
// shape. This is the C++-side counterpart to tools/dry_run_convert.py and is
// the foundation that llama_model_load_mtp_overlay() will build on.
//
// Usage:
//   test-gemma4-mtp-validate /path/to/gemma-4-31B-it-mtp.gguf
//
// Exit code: 0 on success, non-zero with a printed list of failures.

#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct expected_kv {
    const char * key;
    enum gguf_type type;
    // 0 = check presence only; non-zero = check value too
    int64_t expected_value = 0;
    bool check_value = false;
    const char * description = "";
};

struct expected_tensor {
    const char * name;
    std::vector<int64_t> shape;       // expected shape (least-significant dim first, like gguf storage)
    enum ggml_type type;
    const char * description = "";
};

bool kv_present_with_type(const gguf_context * ctx, const char * key, enum gguf_type expected_type, const char ** found_type_name) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return false;
    enum gguf_type actual = gguf_get_kv_type(ctx, id);
    *found_type_name = gguf_type_name(actual);
    return actual == expected_type;
}

bool tensor_present_with_shape(const gguf_context * ctx, const char * name,
                               const std::vector<int64_t> & expected_shape,
                               enum ggml_type expected_type,
                               std::string * err) {
    int64_t id = gguf_find_tensor(ctx, name);
    if (id < 0) {
        *err = "not found";
        return false;
    }
    enum ggml_type actual_type = gguf_get_tensor_type(ctx, id);
    if (actual_type != expected_type) {
        char buf[256];
        snprintf(buf, sizeof(buf), "wrong type: got %s, expected %s",
                 ggml_type_name(actual_type), ggml_type_name(expected_type));
        *err = buf;
        return false;
    }
    // gguf doesn't expose tensor shape directly via this API — we have to load
    // it via ggml. For the validator we accept "size in bytes matches expected
    // product of dims".
    size_t actual_size = gguf_get_tensor_size(ctx, id);
    size_t expected_size = ggml_type_size(expected_type);
    for (auto d : expected_shape) expected_size *= (size_t) d;
    // round up to ggml_blck_size alignment for quantized types (n/a for F16/F32)
    if (actual_size != expected_size) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "size mismatch: got %zu bytes, expected %zu bytes (shape=%s)",
                 actual_size, expected_size,
                 [&]() {
                     static thread_local std::string s;
                     s = "[";
                     for (size_t i = 0; i < expected_shape.size(); ++i) {
                         if (i) s += ", ";
                         s += std::to_string(expected_shape[i]);
                     }
                     s += "]";
                     return s.c_str();
                 }());
        *err = buf;
        return false;
    }
    return true;
}

} // anon

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s /path/to/gemma-4-...-mtp.gguf\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];

    struct gguf_init_params params = {};
    params.no_alloc = true;            // we only inspect, don't load tensor data
    params.ctx      = nullptr;

    gguf_context * gguf = gguf_init_from_file(path, params);
    if (!gguf) {
        std::fprintf(stderr, "FAIL: gguf_init_from_file returned NULL\n");
        return 1;
    }

    std::printf("== %s ==\n", path);
    std::printf("  version       = %u\n", gguf_get_version(gguf));
    std::printf("  alignment     = %zu\n", gguf_get_alignment(gguf));
    std::printf("  n_kv          = %lld\n", (long long) gguf_get_n_kv(gguf));
    std::printf("  n_tensors     = %lld\n", (long long) gguf_get_n_tensors(gguf));
    std::printf("\n");

    int failures = 0;
    auto fail = [&](const char * label, const std::string & detail) {
        std::printf("  FAIL %s: %s\n", label, detail.c_str());
        failures++;
    };
    auto pass = [&](const char * label) {
        std::printf("  ok   %s\n", label);
    };

    // ---- expected metadata ----

    std::printf("== KV metadata ==\n");

    // general
    {
        int64_t id = gguf_find_key(gguf, "general.architecture");
        if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
            fail("general.architecture", "missing or wrong type");
        } else {
            const char * arch = gguf_get_val_str(gguf, id);
            if (std::strcmp(arch, "gemma4") != 0) {
                fail("general.architecture", std::string("expected 'gemma4', got '") + arch + "'");
            } else {
                pass("general.architecture == 'gemma4'");
            }
        }
    }

    // MTP-required keys
    struct mtp_kv_entry {
        const char * key;
        enum gguf_type type;
    };
    const std::array<mtp_kv_entry, 12> required_mtp_kv = {{
        {"gemma4.nextn_predict_layers",                            GGUF_TYPE_UINT32},
        {"gemma4.mtp.hidden_size",                                 GGUF_TYPE_UINT32},
        {"gemma4.mtp.intermediate_size",                           GGUF_TYPE_UINT32},
        {"gemma4.mtp.attention.head_count",                        GGUF_TYPE_UINT32},
        {"gemma4.mtp.attention.head_count_kv",                     GGUF_TYPE_UINT32},
        {"gemma4.mtp.attention.head_dim",                          GGUF_TYPE_UINT32},
        {"gemma4.mtp.attention.global_head_dim",                   GGUF_TYPE_UINT32},
        {"gemma4.mtp.attention.sliding_window",                    GGUF_TYPE_UINT32},
        {"gemma4.mtp.attention.layer_norm_rms_epsilon",            GGUF_TYPE_FLOAT32},
        {"gemma4.mtp.layer_types",                                 GGUF_TYPE_ARRAY},
        {"gemma4.mtp.rope.full.partial_rotary_factor",             GGUF_TYPE_FLOAT32},
        {"gemma4.mtp.rope.sliding.theta_e3",                       GGUF_TYPE_UINT32},
    }};
    for (const auto & e : required_mtp_kv) {
        const char * actual = "absent";
        if (kv_present_with_type(gguf, e.key, e.type, &actual)) {
            int64_t id = gguf_find_key(gguf, e.key);
            std::string val;
            switch (e.type) {
                case GGUF_TYPE_UINT32:  val = std::to_string(gguf_get_val_u32(gguf, id)); break;
                case GGUF_TYPE_FLOAT32: val = std::to_string(gguf_get_val_f32(gguf, id)); break;
                case GGUF_TYPE_ARRAY:   val = std::string("array of size ") +
                                              std::to_string(gguf_get_arr_n(gguf, id)); break;
                default: val = "(?)";
            }
            std::printf("  ok   %-55s = %s\n", e.key, val.c_str());
        } else {
            fail(e.key, std::string("missing or wrong type (got ") + actual + ")");
        }
    }

    // Read nextn_predict_layers for validation below.
    int64_t pl_id = gguf_find_key(gguf, "gemma4.nextn_predict_layers");
    uint32_t n_predict = pl_id < 0 ? 0 : gguf_get_val_u32(gguf, pl_id);

    int64_t hidden_id = gguf_find_key(gguf, "gemma4.mtp.hidden_size");
    uint32_t mtp_hidden = hidden_id < 0 ? 0 : gguf_get_val_u32(gguf, hidden_id);

    int64_t lt_id = gguf_find_key(gguf, "gemma4.mtp.layer_types");
    std::vector<int32_t> layer_types;
    if (lt_id >= 0) {
        size_t n = gguf_get_arr_n(gguf, lt_id);
        const int32_t * data = (const int32_t *) gguf_get_arr_data(gguf, lt_id);
        layer_types.assign(data, data + n);
    }

    // Specific value checks for our 31B drafter:
    if (n_predict != 4)  fail("nextn_predict_layers value", std::string("expected 4, got ") + std::to_string(n_predict));
    if (mtp_hidden != 1024) fail("mtp.hidden_size value", std::string("expected 1024, got ") + std::to_string(mtp_hidden));
    if (layer_types != std::vector<int32_t>{0,0,0,1}) {
        std::string got = "[";
        for (size_t i = 0; i < layer_types.size(); ++i) { if (i) got += ", "; got += std::to_string(layer_types[i]); }
        got += "]";
        fail("mtp.layer_types value", std::string("expected [0,0,0,1], got ") + got);
    } else {
        pass("mtp.layer_types == [0,0,0,1] (3 sliding + 1 full)");
    }

    // ---- expected tensors ----

    std::printf("\n== Tensors ==\n");

    auto check_tensor = [&](const char * name, std::vector<int64_t> shape, enum ggml_type type) {
        std::string err;
        if (tensor_present_with_shape(gguf, name, shape, type, &err)) {
            pass(name);
        } else {
            fail(name, err);
        }
    };

    // Top-level
    check_tensor("mtp.pre_proj.weight",     {10752, 1024}, GGML_TYPE_F16);  // 2*backbone × mtp_hidden
    check_tensor("mtp.post_proj.weight",    { 1024, 5376}, GGML_TYPE_F16);  // mtp_hidden × backbone
    check_tensor("mtp.norm.weight",         { 1024},       GGML_TYPE_F32);
    check_tensor("mtp.embed_tokens.weight", { 1024, 262144}, GGML_TYPE_F16); // tied lm_head

    // Per-block: layers 0..2 use sliding (head_dim=256, q_dim=8192)
    for (int il = 0; il < 3; ++il) {
        char nbuf[128];
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.attn_norm.weight", il);       check_tensor(nbuf, {1024}, GGML_TYPE_F32);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.attn_post_norm.weight", il);  check_tensor(nbuf, {1024}, GGML_TYPE_F32);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.ffn_pre_norm.weight", il);    check_tensor(nbuf, {1024}, GGML_TYPE_F32);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.ffn_post_norm.weight", il);   check_tensor(nbuf, {1024}, GGML_TYPE_F32);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.attn_q.weight", il);          check_tensor(nbuf, {1024, 8192}, GGML_TYPE_F16);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.attn_q_norm.weight", il);     check_tensor(nbuf, {256},  GGML_TYPE_F32);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.attn_output.weight", il);     check_tensor(nbuf, {8192, 1024}, GGML_TYPE_F16);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.ffn_gate.weight", il);        check_tensor(nbuf, {1024, 8192}, GGML_TYPE_F16);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.ffn_up.weight", il);          check_tensor(nbuf, {1024, 8192}, GGML_TYPE_F16);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.ffn_down.weight", il);        check_tensor(nbuf, {8192, 1024}, GGML_TYPE_F16);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.layer_scalar.weight", il);    check_tensor(nbuf, {1},    GGML_TYPE_F32);
    }
    // Layer 3 uses full attention (global_head_dim=512, q_dim=16384)
    {
        const int il = 3;
        char nbuf[128];
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.attn_norm.weight", il);       check_tensor(nbuf, {1024},          GGML_TYPE_F32);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.attn_post_norm.weight", il);  check_tensor(nbuf, {1024},          GGML_TYPE_F32);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.ffn_pre_norm.weight", il);    check_tensor(nbuf, {1024},          GGML_TYPE_F32);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.ffn_post_norm.weight", il);   check_tensor(nbuf, {1024},          GGML_TYPE_F32);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.attn_q.weight", il);          check_tensor(nbuf, {1024, 16384},   GGML_TYPE_F16);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.attn_q_norm.weight", il);     check_tensor(nbuf, {512},           GGML_TYPE_F32);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.attn_output.weight", il);     check_tensor(nbuf, {16384, 1024},   GGML_TYPE_F16);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.ffn_gate.weight", il);        check_tensor(nbuf, {1024, 8192},    GGML_TYPE_F16);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.ffn_up.weight", il);          check_tensor(nbuf, {1024, 8192},    GGML_TYPE_F16);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.ffn_down.weight", il);        check_tensor(nbuf, {8192, 1024},    GGML_TYPE_F16);
        snprintf(nbuf, sizeof(nbuf), "mtp.blk.%d.layer_scalar.weight", il);    check_tensor(nbuf, {1},             GGML_TYPE_F32);
    }

    gguf_free(gguf);

    std::printf("\n== Summary ==\n");
    if (failures == 0) {
        std::printf("ALL CHECKS PASSED\n");
        return 0;
    } else {
        std::printf("%d FAILURE(S)\n", failures);
        return 1;
    }
}
