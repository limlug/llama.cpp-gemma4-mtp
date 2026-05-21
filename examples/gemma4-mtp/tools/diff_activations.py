#!/usr/bin/env python3
"""Compare HF reference activations to C++ harness output.

Reads:
  reference_activations.npz   (HF, written by reference_probe.py)
  acts_cpp.bin + acts_cpp.json (C++, written by test-gemma4-mtp-ref-diff)

For each label present in both: reports shape, max-abs-error, mean-abs-error,
cosine similarity, and argmax agreement (for logits).
"""
import argparse, json
from pathlib import Path
import numpy as np


def load_cpp(out_dir: Path):
    manifest = json.loads((out_dir / "acts_cpp.json").read_text())
    blob = (out_dir / "acts_cpp.bin").read_bytes()
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
    arg_a = int(a.argmax())
    arg_b = int(b.argmax())
    ok = mxe < 1e-3
    flag = "✓" if ok else ("≈" if mxe < 1e-1 else "✗")
    return (f"  {flag} {label:40s} max_abs={mxe:.4e}  mean_abs={mae:.4e}  "
            f"cos={cos:.6f}  argmax_hf={arg_a} cpp={arg_b}{' ✓' if arg_a == arg_b else ' ✗'}")


def reconstruct_masked_logits(cpp: dict, vocab: int) -> np.ndarray | None:
    """If the C++ side emitted sparse masked_embedding outputs, scatter them
    into a full-vocab logits vector using HF's mask_value formula:
      logits[v] = selected_logits[i] if v in selected_indices else min - 1.0
    Returns None if the sparse outputs aren't present.
    """
    if "selected_logits" not in cpp or "selected_indices" not in cpp:
        return None
    sl  = np.asarray(cpp["selected_logits"]).reshape(-1).astype(np.float32)
    si  = np.asarray(cpp["selected_indices"]).reshape(-1).astype(np.float32).astype(np.int64)
    mv  = float(sl.min()) - 1.0
    out = np.full((vocab,), mv, dtype=np.float32)
    # If duplicates in si (HF allows but the last write wins via scatter_), use
    # numpy's "later index wins" by using assignment in order.
    out[si] = sl
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf",  default="/tmp/reference_activations.npz")
    p.add_argument("--cpp", default="/tmp")
    args = p.parse_args()

    hf  = np.load(args.hf)
    cpp = load_cpp(Path(args.cpp))

    # If the C++ side emitted sparse masked_embedding outputs, reconstruct the
    # full-vocab logits and inject as "logits_masked" so it's diffed against HF.
    if "selected_logits" in cpp:
        vocab = int(np.asarray(hf["logits"]).reshape(-1).shape[0])
        reconstructed = reconstruct_masked_logits(cpp, vocab)
        if reconstructed is not None:
            cpp["logits_masked"] = reconstructed
            print(f"[reconstructed logits_masked from sparse outputs: vocab={vocab}, K={len(np.asarray(cpp['selected_logits']).reshape(-1))}]")

    print(f"HF labels:  {len(hf.files)}")
    print(f"CPP labels: {len(cpp)}")
    print(f"Common:     {sorted(set(hf.files) & set(cpp))}")
    print()
    print("Per-label diff:")
    for label in sorted(cpp):
        # Map our synthetic "logits_masked" label to HF's "logits"
        hf_label = "logits" if label == "logits_masked" else label
        if hf_label not in hf:
            print(f"  ? {label:40s} not in HF")
            continue
        # HF may have an extra leading batch dim — squeeze to compare
        a = np.asarray(hf[hf_label])
        b = np.asarray(cpp[label])
        # Squeeze [1, ..., V] → [V] for logits-style comparison
        a = np.squeeze(a)
        b = np.squeeze(b)
        # For logits, HF is per-position; pick the last position.
        if hf_label == "logits" and a.ndim == 2 and b.ndim == 1:
            a = a[-1]
        print(diff(label, a, b))


if __name__ == "__main__":
    main()
