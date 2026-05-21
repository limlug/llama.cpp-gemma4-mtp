// tests/test_gemma4_mtp_reference.cpp
//
// Numerical-reference smoke test for the Gemma4 MTP graph.
//
// Loads tools/reference_activations.json, builds a llama_context against a
// Gemma4 + MTP-overlay GGUF, runs one MTP step with the same RNG-derived
// inputs, and compares intermediate activations + final logits within a
// tolerance.
//
// Build:
//   add to tests/CMakeLists.txt:
//     llama_target_and_test(test-gemma4-mtp-reference.cpp)
//
// Run:
//   ./build/bin/test-gemma4-mtp-reference \
//       --base gemma4-31b.gguf \
//       --mtp  gemma4-31b-mtp.gguf \
//       --ref  tools/reference_activations.json
//
// Pass criteria (relative tolerance):
//   - pre_projection output:  rtol < 1e-3 (fp32) / 1e-2 (fp16)
//   - L0..L3 q_proj output :  rtol < 1e-3 / 5e-2
//   - L0..L3 o_proj output :  rtol < 5e-3 / 5e-2
//   - model.norm output    :  rtol < 1e-3 / 5e-2
//   - post_projection      :  rtol < 5e-3 / 5e-2
//   - logits argmax pos 0  :  must equal 89004 (exact)
//
// If any check fails, the test prints the first 16 elements of both expected
// and actual tensors, and the max absolute / relative diff.

#include "llama.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// A minimal JSON reader for our specific schema. (We avoid pulling a full
// JSON dep; reference_activations.json has a known structure.)
//
// For robustness in production, replace with nlohmann/json or similar.
struct ref_activation {
    std::vector<int64_t> shape;
    std::vector<float>   first16;
    float                mean = 0.0f;
    float                std_ = 0.0f;
    float                abs_max = 0.0f;
};

struct reference_data {
    int batch = 1;
    int seq   = 4;
    int kv_len = 8;
    std::vector<float> inputs_embeds_first16;
    std::vector<float> shared_kv_sliding_K_first16;
    std::vector<float> shared_kv_full_K_first16;

    int backbone_hidden_size = 5376;
    int mtp_hidden           = 1024;
    int n_layers             = 4;
    int n_attn_heads         = 32;
    int n_kv_heads           = 16;
    int head_dim_swa         = 256;
    int head_dim_full        = 512;

    std::vector<int64_t> logits_shape;
    std::vector<float>   logits_first16_pos0;
    int                  logits_argmax_pos0 = -1;
    std::vector<int64_t> last_hidden_state_shape;
    std::vector<float>   last_hidden_state_first16;

    // keyed by label, e.g. "pre_projection", "L0.q_proj", ...
    std::vector<std::pair<std::string, ref_activation>> activations;
};

// (parser implementation elided; see test_helper_parse_ref.h)
reference_data load_reference(const std::string & path);


static bool close_enough(float a, float b, float rtol, float atol) {
    const float d = std::fabs(a - b);
    return d <= atol || d <= rtol * std::max(std::fabs(a), std::fabs(b));
}

static bool compare_first16(
        const char * label,
        const float * expected,
        const float * actual,
        size_t n,
        float rtol,
        float atol)
{
    bool ok = true;
    float max_abs = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (!close_enough(expected[i], actual[i], rtol, atol)) {
            ok = false;
        }
        const float d = std::fabs(expected[i] - actual[i]);
        max_abs = std::max(max_abs, d);
        if (std::fabs(expected[i]) > 1e-6f) {
            max_rel = std::max(max_rel, d / std::fabs(expected[i]));
        }
    }
    if (!ok) {
        std::fprintf(stderr, "FAIL %s: max_abs=%.4g max_rel=%.4g\n", label, max_abs, max_rel);
        std::fprintf(stderr, "  expected: ");
        for (size_t i = 0; i < n; ++i) std::fprintf(stderr, "%+.4g ", expected[i]);
        std::fprintf(stderr, "\n  actual:   ");
        for (size_t i = 0; i < n; ++i) std::fprintf(stderr, "%+.4g ", actual[i]);
        std::fprintf(stderr, "\n");
    } else {
        std::fprintf(stderr, "ok   %s\n", label);
    }
    return ok;
}


int main(int argc, char ** argv) {
    std::string base_path, mtp_path, ref_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--base" && i + 1 < argc) base_path = argv[++i];
        else if (a == "--mtp"  && i + 1 < argc) mtp_path  = argv[++i];
        else if (a == "--ref"  && i + 1 < argc) ref_path  = argv[++i];
    }
    if (base_path.empty() || mtp_path.empty() || ref_path.empty()) {
        std::fprintf(stderr, "usage: %s --base BASE.gguf --mtp MTP.gguf --ref reference_activations.json\n", argv[0]);
        return 2;
    }

    const reference_data ref = load_reference(ref_path);
    std::fprintf(stderr, "Loaded %zu activation records from %s\n",
                 ref.activations.size(), ref_path.c_str());

    // -- TODO: load base + MTP gguf, build llama_context with type=MTP --
    //
    // llama_model_params mp = llama_model_default_params();
    // llama_model * model = llama_model_load_from_file(base_path.c_str(), mp);
    // // load MTP overlay onto model: pseudo-API
    // llama_model_load_mtp_overlay(model, mtp_path.c_str());
    //
    // llama_context_params cp = llama_context_default_params();
    // cp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
    // cp.embeddings_pre_norm = true;
    // cp.export_shared_kv    = true;
    //
    // llama_context * ctx = llama_init_from_model(model, cp);

    // -- TODO: bind inputs from reference data --
    //
    // Convert ref.inputs_embeds_first16 etc. into the MTP graph's input tensors
    // (h_in, tokens, shared_K_swa, shared_V_swa, shared_K_full, shared_V_full).
    //
    // The full input vectors are NOT in the JSON (only first16 samples). To do
    // a true numerical reference test, we need the FULL input tensors. Two
    // options:
    //   A) Add full input dump to reference_probe.py and re-run.
    //   B) Reproduce the same RNG seeding in C++ and regenerate inputs.
    //
    // Option B is cleaner because (1) the test stays self-contained, (2) it
    // verifies the test harness's seeding logic too. Python's torch.randn with
    // generator seed 0 is reproducible but not bit-exact across platforms;
    // safer to add to the probe a full-dump option and ship the inputs as a
    // .bin file alongside the JSON.
    //
    // For now: extend reference_probe.py with `--dump-full` to write inputs as
    // float32 binaries, and load those here.

    // -- TODO: run forward, capture intermediates, compare --
    //
    // For each activation in ref.activations, extract the corresponding
    // tensor from the result graph and compare with compare_first16().

    std::fprintf(stderr, "\n!! test scaffold only — full implementation requires:\n");
    std::fprintf(stderr, "   1) ./convert_hf_to_gguf.py --mtp emits gemma-4-31B-it-mtp.gguf\n");
    std::fprintf(stderr, "   2) llama_model API to load the MTP overlay\n");
    std::fprintf(stderr, "   3) reference_probe.py --dump-full to emit reference_inputs.bin\n");
    std::fprintf(stderr, "   4) implement the binding+compare loop above\n");
    return 1;
}


// -------------------------------------------------------------------------
// load_reference: minimal JSON parser for our schema.
// -------------------------------------------------------------------------
//
// Implementation kept inline for the scaffold. In production, swap for a
// real JSON library. This parser uses naive substring search and assumes
// well-formed input produced by tools/reference_probe.py.

#include <algorithm>
#include <cctype>

static std::string slurp(const std::string & p) {
    std::ifstream f(p);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

static std::vector<float> parse_float_array(const std::string & s, size_t & off) {
    std::vector<float> out;
    while (off < s.size() && s[off] != '[') ++off;
    if (off == s.size()) return out;
    ++off;
    while (off < s.size() && s[off] != ']') {
        while (off < s.size() && (std::isspace((unsigned char)s[off]) || s[off] == ',')) ++off;
        if (off < s.size() && s[off] != ']') {
            char * end = nullptr;
            float v = std::strtof(&s[off], &end);
            out.push_back(v);
            off = (size_t)(end - s.data());
        }
    }
    if (off < s.size()) ++off;
    return out;
}

reference_data load_reference(const std::string & path) {
    reference_data r;
    const std::string buf = slurp(path);
    auto field = [&](const std::string & key) -> size_t {
        size_t pos = buf.find("\"" + key + "\"");
        return pos == std::string::npos ? buf.size() : pos + key.size() + 2;
    };
    size_t p;
    p = field("inputs_embeds_first16");        r.inputs_embeds_first16        = parse_float_array(buf, p);
    p = field("shared_kv_sliding_K_first16");  r.shared_kv_sliding_K_first16  = parse_float_array(buf, p);
    p = field("shared_kv_full_K_first16");     r.shared_kv_full_K_first16     = parse_float_array(buf, p);
    p = field("logits_first16_pos0");          r.logits_first16_pos0          = parse_float_array(buf, p);
    p = field("last_hidden_state_first16");    r.last_hidden_state_first16    = parse_float_array(buf, p);
    // logits_argmax_pos0
    p = field("logits_argmax_pos0");
    while (p < buf.size() && !std::isdigit((unsigned char)buf[p]) && buf[p] != '-') ++p;
    r.logits_argmax_pos0 = (int) std::strtol(&buf[p], nullptr, 10);
    return r;
}
