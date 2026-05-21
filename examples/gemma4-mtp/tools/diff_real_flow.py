#!/usr/bin/env python3
"""Compare C++ mini-spec dump against HF real_flow_probe outputs.

C++ side: /tmp/acts_cpp_real.bin + .json (written by mini-spec when env
MINI_SPEC_DUMP_DIR is set).

HF side: /tmp/real_acts.npz + /tmp/real_inputs.npz (written by real_flow_probe.py).

Diff: for each common label, report cosine and argmax agreement.
"""
import argparse, json
from pathlib import Path
import numpy as np


def load_cpp(out_dir: Path):
    manifest = json.loads((out_dir / "acts_cpp_real.json").read_text())
    blob = (out_dir / "acts_cpp_real.bin").read_bytes()
    out = {}
    for k, m in manifest.items():
        arr = np.frombuffer(blob, dtype=np.float32, count=m["n_bytes"] // 4,
                            offset=m["offset"]).reshape(m["shape"])
        out[k] = arr
    return out


def diff(label, a, b):
    a = np.asarray(a).reshape(-1).astype(np.float64)
    b = np.asarray(b).reshape(-1).astype(np.float64)
    if a.shape != b.shape:
        return f"  {label:40s} SHAPE MISMATCH: hf={a.shape} cpp={b.shape}"
    delta = a - b
    mae = float(np.abs(delta).mean())
    mxe = float(np.abs(delta).max())
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    arg_a = int(a.argmax()); arg_b = int(b.argmax())
    flag = "✓" if mxe < 1e-3 else ("≈" if mxe < 1e-1 else "✗")
    return (f"  {flag} {label:40s} max_abs={mxe:.4e}  mean_abs={mae:.4e}  "
            f"cos={cos:.6f}  argmax_hf={arg_a} cpp={arg_b}{' ✓' if arg_a == arg_b else ' ✗'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpp", default="/tmp")
    p.add_argument("--hf-acts", default="/tmp/real_acts.npz")
    p.add_argument("--hf-inp",  default="/tmp/real_inputs.npz")
    args = p.parse_args()

    cpp = load_cpp(Path(args.cpp))
    hf_acts = np.load(args.hf_acts)
    hf_inp  = np.load(args.hf_inp)

    print(f"C++ labels:  {len(cpp)}")
    print(f"HF acts:     {len(hf_acts.files)}")
    print(f"HF inputs:   {len(hf_inp.files)}")

    print("\n=== Inputs comparison (C++ derived from base GGUF vs HF derived from base PT) ===")
    if "h_t" in cpp:
        a = np.asarray(hf_inp["h_t"]).reshape(-1)
        b = np.asarray(cpp["h_t"]).reshape(-1)
        print(diff("h_t (base last_hidden_state)", a, b))

    print("\n=== Drafter activations ===")
    # Reconstruct full masked logits if sparse outputs present
    if "selected_logits" in cpp and "selected_indices" in cpp:
        sl  = np.asarray(cpp["selected_logits"]).reshape(-1)
        si  = np.asarray(cpp["selected_indices"]).reshape(-1).astype(np.int64)
        vocab = int(np.asarray(hf_acts["logits"]).reshape(-1).shape[0])
        mv = float(sl.min()) - 1.0
        rec = np.full((vocab,), mv, dtype=np.float32)
        rec[si] = sl
        cpp["logits_masked"] = rec

    common_labels = ["pre_projection", "L0.input_layernorm", "L0.q_proj", "L0.q_norm",
                     "L0.attn_out_pre_o_proj", "L0.o_proj", "L0.out",
                     "model.norm", "post_projection", "logits_masked"]
    for label in common_labels:
        hf_label = "logits" if label == "logits_masked" else label
        if label not in cpp or hf_label not in hf_acts:
            print(f"  ? {label:40s} cpp={label in cpp} hf={hf_label in hf_acts.files}")
            continue
        a = np.squeeze(np.asarray(hf_acts[hf_label]))
        b = np.squeeze(np.asarray(cpp[label]))
        if hf_label == "logits" and a.ndim == 2 and b.ndim == 1:
            a = a[-1]
        print(diff(label, a, b))


if __name__ == "__main__":
    main()
