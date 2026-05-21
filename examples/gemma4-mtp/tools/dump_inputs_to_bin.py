#!/usr/bin/env python3
"""Convert reference_inputs.npz + reference_activations.npz to flat .bin files
+ manifest.json, so the C++ harness doesn't need an npz reader.

Layout written to --out-dir:
  inputs.bin        # all input tensors concatenated, fp32
  inputs.json       # {label: {offset, n_floats, shape, dtype}}
  acts.bin          # all activation tensors concatenated, fp32
  acts.json         # same structure
"""
import argparse, json
from pathlib import Path
import numpy as np


def dump(npz_path: Path, bin_path: Path, json_path: Path):
    z = np.load(npz_path)
    manifest = {}
    offset = 0
    with open(bin_path, "wb") as f:
        for key in z.files:
            arr = np.asarray(z[key])
            # token_ids / position_ids / attention_mask are int — keep their dtype
            if arr.dtype != np.float32 and arr.dtype.kind == "f":
                arr = arr.astype(np.float32)
            elif arr.dtype.kind == "i":
                arr = arr.astype(np.int32)
            data = arr.tobytes()
            f.write(data)
            manifest[key] = {
                "offset": offset,
                "n_bytes": len(data),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
            offset += len(data)
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  {bin_path}: {offset:,} bytes, {len(manifest)} tensors")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="/tmp")
    p.add_argument("--out-dir", default="/tmp")
    args = p.parse_args()
    inp = Path(args.in_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump(inp / "reference_inputs.npz",      out / "inputs.bin", out / "inputs.json")
    dump(inp / "reference_activations.npz", out / "acts.bin",   out / "acts.json")


if __name__ == "__main__":
    main()
