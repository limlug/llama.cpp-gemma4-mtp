// tests/test-gemma4-mtp-ref-diff.cpp
//
// Reference-diff harness: replays the inputs captured by tools/reference_probe.py
// through our C++ MTP forward, dumps logits + intermediates to acts_cpp.bin,
// so tools/diff_activations.py can compare layer-by-layer to HF reference.
//
// Inputs read from {inputs.bin, inputs.json}: h_part [B, S, bh] (random),
//   token_ids [B, S] (int32), K_swa/V_swa/K_full/V_full (F32, converted to
//   F16 here to match the MTP context's input tensor type).
// Outputs written to {acts_cpp.bin, acts_cpp.json}: logits per step.
//
// Usage:
//   test-gemma4-mtp-ref-diff BASE.gguf OVERLAY.gguf INPUTS_DIR OUT_DIR

#include "llama.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

struct Entry {
    size_t offset = 0;
    size_t n_bytes = 0;
    std::vector<int64_t> shape;
    std::string dtype;
};

// Tiny ad-hoc parser for the manifest JSON written by dump_inputs_to_bin.py.
// The format is a flat object: { "KEY": {"offset": N, "n_bytes": N, "shape":
// [...], "dtype": "..."} , ... }. Robust enough for our controlled writer.
static std::map<std::string, Entry> parse_manifest(const std::string & path) {
    std::ifstream in(path);
    std::string j((std::istreambuf_iterator<char>(in)), {});
    std::map<std::string, Entry> out;
    size_t p = 0;
    while (true) {
        // find next '"KEY":'
        size_t k0 = j.find('"', p);
        if (k0 == std::string::npos) break;
        size_t k1 = j.find('"', k0 + 1);
        if (k1 == std::string::npos) break;
        std::string key = j.substr(k0 + 1, k1 - k0 - 1);
        // Only take top-level keys (skip ones inside an inner object). Crude
        // heuristic: peek the char after the next ':' — if it's '{' it's a
        // value object (good); else skip.
        size_t colon = j.find(':', k1);
        if (colon == std::string::npos) break;
        size_t nb = colon + 1;
        while (nb < j.size() && (j[nb] == ' ' || j[nb] == '\n' || j[nb] == '\t')) ++nb;
        if (nb >= j.size() || j[nb] != '{') { p = colon + 1; continue; }
        size_t end = j.find('}', nb);
        if (end == std::string::npos) break;
        std::string body = j.substr(nb, end - nb + 1);
        Entry e;
        auto pull_int = [&](const char * name, size_t & dst) -> bool {
            std::string pat = std::string("\"") + name + "\":";
            size_t pos = body.find(pat);
            if (pos == std::string::npos) return false;
            pos += pat.size();
            while (pos < body.size() && body[pos] == ' ') ++pos;
            dst = (size_t) std::strtoull(body.c_str() + pos, nullptr, 10);
            return true;
        };
        pull_int("offset", e.offset);
        pull_int("n_bytes", e.n_bytes);
        // shape
        size_t sh_pos = body.find("\"shape\":");
        if (sh_pos != std::string::npos) {
            size_t lb = body.find('[', sh_pos);
            size_t rb = body.find(']', lb);
            std::string s = body.substr(lb + 1, rb - lb - 1);
            size_t q = 0;
            while (q < s.size()) {
                while (q < s.size() && (s[q] == ' ' || s[q] == ',')) ++q;
                if (q >= s.size()) break;
                e.shape.push_back(std::strtoll(s.c_str() + q, nullptr, 10));
                while (q < s.size() && s[q] != ',') ++q;
            }
        }
        // dtype
        size_t dt_pos = body.find("\"dtype\":");
        if (dt_pos != std::string::npos) {
            size_t q0 = body.find('"', dt_pos + 8);
            size_t q1 = body.find('"', q0 + 1);
            e.dtype = body.substr(q0 + 1, q1 - q0 - 1);
        }
        out[key] = std::move(e);
        p = end + 1;
    }
    return out;
}

static std::vector<uint8_t> read_bin(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { std::fprintf(stderr, "fail to open %s\n", path.c_str()); std::exit(2); }
    auto n = in.tellg(); in.seekg(0);
    std::vector<uint8_t> buf(n);
    in.read((char *) buf.data(), n);
    return buf;
}

// F32 → F16 (IEEE half) for K/V binding. Match ggml's GGML_FP32_TO_FP16 layout.
static uint16_t f32_to_f16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    uint32_t sign = (u >> 31) & 0x1;
    int32_t exp  = ((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3ff;
    if (exp >= 31) return (uint16_t)((sign << 15) | (0x1f << 10) | (((u >> 23) & 0xff) == 0xff ? mant : 0));
    if (exp <= 0)  return (uint16_t)(sign << 15);
    return (uint16_t)((sign << 15) | (exp << 10) | mant);
}

static void f32_to_f16_buf(const float * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s BASE.gguf OVERLAY.gguf INPUTS_DIR OUT_DIR\n", argv[0]);
        return 2;
    }
    const std::string base_path     = argv[1];
    const std::string overlay_path  = argv[2];
    const std::string inputs_dir    = argv[3];
    const std::string out_dir       = argv[4];

    auto manifest = parse_manifest(inputs_dir + "/inputs.json");
    auto blob     = read_bin(inputs_dir + "/inputs.bin");
    auto get = [&](const char * key) -> std::pair<const uint8_t *, const Entry *> {
        auto it = manifest.find(key);
        if (it == manifest.end()) { std::fprintf(stderr, "missing key '%s'\n", key); std::exit(2); }
        return { blob.data() + it->second.offset, &it->second };
    };

    auto [h_data,  h_e]  = get("h_part");
    auto [tok_data,tok_e]= get("token_ids");
    auto [ks_data, ks_e] = get("K_swa");
    auto [vs_data, vs_e] = get("V_swa");
    auto [kf_data, kf_e] = get("K_full");
    auto [vf_data, vf_e] = get("V_full");

    std::printf("Loaded inputs:\n");
    std::printf("  h_part:    shape=[%lld,%lld,%lld] dtype=%s\n",
        (long long) h_e->shape[0], (long long) h_e->shape[1], (long long) h_e->shape[2],
        h_e->dtype.c_str());
    std::printf("  token_ids: shape=[%lld,%lld] dtype=%s\n",
        (long long) tok_e->shape[0], (long long) tok_e->shape[1], tok_e->dtype.c_str());
    std::printf("  K_swa:     shape=[%lld,%lld,%lld,%lld] dtype=%s\n",
        (long long) ks_e->shape[0], (long long) ks_e->shape[1],
        (long long) ks_e->shape[2], (long long) ks_e->shape[3], ks_e->dtype.c_str());

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(base_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "load base failed\n"); return 1; }
    int rc = llama_model_load_mtp_overlay(model, overlay_path.c_str());
    if (rc != 0) { std::fprintf(stderr, "overlay rc=%d\n", rc); return 1; }

    llama_context_params cp = llama_context_default_params();
    // n_batch=n_ubatch=1: MTP decode is single-token, and the masked_embedding
    // path's ggml_get_rows on embed_tokens produces large intermediates at
    // higher n_tokens that crash sched_reserve. Smaller batch = smaller probe.
    cp.n_ctx = 256; cp.n_batch = 1; cp.n_ubatch = 1; cp.n_threads = 4;
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    llama_context * ctx = llama_init_from_model(model, cp);

    // Convert K/V from F32 → F16 for binding (MTP context expects F16).
    // Note: HF probe captures n_kv=KV_LEN (e.g. 8) rows. The MTP context's
    // shared_K/V tensors are sized to cparams.n_ctx as the upper bound. The
    // binding mechanism asserts exact size match (see process_ubatch), so we
    // must pad zero-rows out to n_ctx rows. For the diff we use cparams.n_ctx
    // small enough that this padding is cheap; the real K/V is at rows [0..KV_LEN).
    auto convert_and_pad = [&](const uint8_t * src, size_t src_bytes,
                                int64_t hd, int64_t n_kv_heads, int64_t kv_len) -> std::vector<uint16_t> {
        // Source layout: [batch=1, n_kv_heads, kv_len, hd] (F32).
        // MTP graph tensor: [hd, n_kv_heads, n_ctx] F16 (column-major in ggml convention).
        // For the diff harness we just need the first kv_len rows along n_ctx populated.
        const int64_t n_ctx_pad = (int64_t) cp.n_ctx;
        std::vector<uint16_t> dst((size_t)(hd * n_kv_heads * n_ctx_pad), 0);
        const float * sf = (const float *) src;
        // Transpose src [B=1, KV_H, KV_L, HD] → dst [HD, KV_H, n_ctx_pad]
        for (int64_t h = 0; h < n_kv_heads; ++h) {
            for (int64_t t = 0; t < kv_len; ++t) {
                for (int64_t d = 0; d < hd; ++d) {
                    int64_t src_idx = ((0 * n_kv_heads + h) * kv_len + t) * hd + d;
                    int64_t dst_idx = (t * n_kv_heads + h) * hd + d;
                    dst[dst_idx] = f32_to_f16(sf[src_idx]);
                }
            }
        }
        return dst;
    };
    const int64_t HD_SWA  = ks_e->shape[3];
    const int64_t HD_FULL = kf_e->shape[3];
    const int64_t N_KV_H_SWA  = ks_e->shape[1];
    const int64_t N_KV_H_FULL = kf_e->shape[1];  // 31B has fewer KV heads for FULL
    const int64_t KV_LEN  = ks_e->shape[2];
    std::vector<uint16_t> K_swa_f16  = convert_and_pad(ks_data, ks_e->n_bytes, HD_SWA,  N_KV_H_SWA,  KV_LEN);
    std::vector<uint16_t> V_swa_f16  = convert_and_pad(vs_data, vs_e->n_bytes, HD_SWA,  N_KV_H_SWA,  KV_LEN);
    std::vector<uint16_t> K_full_f16 = convert_and_pad(kf_data, kf_e->n_bytes, HD_FULL, N_KV_H_FULL, KV_LEN);
    std::vector<uint16_t> V_full_f16 = convert_and_pad(vf_data, vf_e->n_bytes, HD_FULL, N_KV_H_FULL, KV_LEN);
    llama_set_input_tensor(ctx, "mtp_shared_K_swa",  K_swa_f16.data(),  K_swa_f16.size()  * sizeof(uint16_t));
    llama_set_input_tensor(ctx, "mtp_shared_V_swa",  V_swa_f16.data(),  V_swa_f16.size()  * sizeof(uint16_t));
    llama_set_input_tensor(ctx, "mtp_shared_K_full", K_full_f16.data(), K_full_f16.size() * sizeof(uint16_t));
    llama_set_input_tensor(ctx, "mtp_shared_V_full", V_full_f16.data(), V_full_f16.size() * sizeof(uint16_t));

    // Attention mask: [kv_max, n_tokens]. 0 for [0..kv_len), -INF for the rest,
    // so padded K positions don't leak into the softmax.
    const int64_t KV_MAX = (int64_t) cp.n_ctx;
    const int64_t N_TOK  = 1;  // single MTP step
    std::vector<float> mask((size_t)(KV_MAX * N_TOK), -std::numeric_limits<float>::infinity());
    for (int64_t t = 0; t < N_TOK; ++t)
        for (int64_t k = 0; k < KV_LEN; ++k)
            mask[(size_t)(t * KV_MAX + k)] = 0.0f;
    llama_set_input_tensor(ctx, "mtp_attn_mask", mask.data(), mask.size() * sizeof(float));

    // Bind the constant 1.0 used by masked_embedding's mask_value calculation.
    // Harmless to bind on contexts that don't use it (binding is ignored if
    // the tensor wasn't declared in the graph).
    const float one_val = 1.0f;
    llama_set_input_tensor(ctx, "mtp_const_one", &one_val, sizeof(float));

    // Bind h_part as mtp_h_input. The graph_mtp expects [backbone_dim, n_tokens]
    // F32 (column-major in ggml's convention). Our HF probe stores [B, S, bh]
    // row-major so for B=1, S=1 (single-step diff) the layout matches.
    const int64_t B  = h_e->shape[0];
    const int64_t S  = h_e->shape[1];
    const int64_t BH = h_e->shape[2];
    std::printf("Binding h_part: B=%lld S=%lld BH=%lld → mtp_h_input (n_tokens=%lld)\n",
                (long long) B, (long long) S, (long long) BH, (long long) (B * S));
    // For now run S=1 (last position). Take the last-position slice.
    const float * h_last = ((const float *) h_data) + (size_t)(S - 1) * BH;
    llama_set_input_tensor(ctx, "mtp_h_input", h_last, BH * sizeof(float));

    // Run MTP decode with the LAST token id.
    const int32_t * tok = (const int32_t *) tok_data;
    int32_t last_tok = tok[S - 1];
    std::printf("Decoding MTP with token_id=%d\n", (int) last_tok);

    llama_batch b = llama_batch_init(1, 0, 1);
    b.token[0] = last_tok;
    b.pos[0] = 0;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.logits[0] = 1;
    b.n_tokens = 1;
    int dr = llama_decode(ctx, b);
    if (dr != 0) { std::fprintf(stderr, "decode failed rc=%d\n", dr); return 1; }

    // Dump logits to acts_cpp.bin + acts_cpp.json (single entry "logits").
    const float * logits = llama_get_logits(ctx);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int argmax = 0; float maxv = logits[0];
    for (uint32_t i = 1; i < n_vocab; ++i) if (logits[i] > maxv) { maxv = logits[i]; argmax = i; }
    std::printf("logits: n_vocab=%u argmax=%d (logit=%.4f)\n", n_vocab, argmax, maxv);

    // Write logits + every debug tap to acts_cpp.bin with a manifest.
    std::ofstream bf(out_dir + "/acts_cpp.bin", std::ios::binary);
    std::ofstream jf(out_dir + "/acts_cpp.json");
    jf << "{\n";
    size_t off = 0;
    auto emit = [&](const std::string & name, const float * data, size_t n_floats, bool first) {
        size_t nb = n_floats * sizeof(float);
        bf.write((const char *) data, (std::streamsize) nb);
        if (!first) jf << ",\n";
        jf << "  \"" << name << "\": {\"offset\": " << off
           << ", \"n_bytes\": " << nb
           << ", \"shape\": [" << n_floats << "], \"dtype\": \"float32\"}";
        off += nb;
    };
    emit("logits", logits, n_vocab, true);
    int n_taps = llama_get_dbg_tap_count(ctx);
    std::printf("Dumping %d dbg taps:\n", n_taps);
    for (int i = 0; i < n_taps; ++i) {
        const char * name = llama_get_dbg_tap_name(ctx, i);
        const float * data = llama_get_dbg_tap_data(ctx, name);
        size_t nf = llama_get_dbg_tap_size(ctx, name);
        std::printf("  %-32s n_floats=%zu\n", name, nf);
        if (data && nf > 0) emit(name, data, nf, false);
    }
    jf << "\n}\n";
    std::printf("Wrote %s/acts_cpp.bin (%zu bytes) + acts_cpp.json\n",
                out_dir.c_str(), off);

    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
