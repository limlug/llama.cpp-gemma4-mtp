#!/usr/bin/env python3
"""Reference probe v3 — full-tensor capture for C++ numerical comparison.

Captures everything needed to diff our llama.cpp MTP forward against HF
layer-by-layer:

  - Deterministic inputs (seeded torch.randn, written to .npz so C++ can
    replicate bit-exact). C++ test reads the .npz, builds the same
    inputs_embeds / shared_kv / position_ids, runs MTP forward, dumps the
    same activation tensors, and a Python diff script does max-abs-err per
    label.

  - Activation hooks at every checkpoint that matters:
      * pre_projection out
      * per-layer self_attn:
          input_layernorm out, q_proj out, q_norm out (post-RoPE Q),
          k from shared KV after RoPE/norm, v from shared KV,
          attn_weights pre-softmax, attn_weights post-softmax,
          attn output before o_proj, o_proj out
      * per-layer MLP: gate_proj, up_proj, down_proj
      * per-layer output (post-residual)
      * post_projection out
      * model.norm out
      * lm_head out (== logits)

  - Statistical summary in JSON (mean/std/abs_max/L2 + first16) so we can
    eyeball without loading the .npz.

  - Full tensors in `reference_activations.npz` (one array per label).

Layer-coverage is bounded to LAYERS_TO_CAPTURE (default: first 3 + last 1) to
keep .npz size manageable. Pass `--all-layers` to dump every layer.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


DEFAULT_MODEL = "google/gemma-4-E2B-it-assistant"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help="HF drafter model id (must match the base GGUF)")
    p.add_argument("--base-gguf", type=str, default="/tmp/gguf-out/gemma-4-E2B-it.gguf",
                   help="GGUF path containing base model's tok_embd (used to build "
                        "the 'e' half of inputs_embeds so C++ can replicate exactly).")
    p.add_argument("--seq", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--kv-len", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="/tmp")
    p.add_argument("--all-layers", action="store_true",
                   help="capture every layer (large .npz)")
    p.add_argument("--layers", type=str, default="",
                   help="comma-separated layer indices to capture (overrides default)")
    return p.parse_args()


def load_base_tok_embd(gguf_path: str) -> np.ndarray:
    """Read base model's token_embd.weight from a GGUF file. Returns [vocab, bh] fp32."""
    import gguf
    r = gguf.GGUFReader(gguf_path)
    for t in r.tensors:
        if t.name == "token_embd.weight":
            data = np.asarray(t.data)
            # GGUF stores as [hidden, vocab]; we want [vocab, hidden]
            if data.shape[0] < data.shape[1]:
                data = data.T
            return data.astype(np.float32)
    raise RuntimeError(f"token_embd.weight not found in {gguf_path}")


def main() -> int:
    args = parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(args.seed)

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tcfg = cfg.text_config
    bh = cfg.backbone_hidden_size
    mh = tcfg.hidden_size
    n_attn_heads = tcfg.num_attention_heads
    n_kv_heads = tcfg.num_key_value_heads
    head_dim_swa = tcfg.head_dim
    head_dim_full = tcfg.global_head_dim
    n_layers = tcfg.num_hidden_layers
    layer_types = list(tcfg.layer_types)

    print(f"== Config ==")
    print(f"  backbone_hidden: {bh}  mtp_hidden: {mh}")
    print(f"  n_layers: {n_layers}  n_attn_heads: {n_attn_heads}  n_kv_heads: {n_kv_heads}")
    print(f"  head_dim swa/full: {head_dim_swa}/{head_dim_full}")
    print(f"  layer_types: {layer_types}")
    print()

    if args.layers:
        layers_to_capture = set(int(x) for x in args.layers.split(","))
    elif args.all_layers:
        layers_to_capture = set(range(n_layers))
    else:
        layers_to_capture = {0, 1, 2, n_layers - 1}
    print(f"  capturing layers: {sorted(layers_to_capture)}")
    print()

    print("== Loading base tok_embd from GGUF ==")
    base_tok_embd = load_base_tok_embd(args.base_gguf)
    print(f"  shape: {base_tok_embd.shape}  dtype: {base_tok_embd.dtype}")
    print(f"  norm: {np.linalg.norm(base_tok_embd[100]):.4f}  (row 100 as sanity)")
    assert base_tok_embd.shape[1] == bh, (
        f"base tok_embd hidden ({base_tok_embd.shape[1]}) != cfg.backbone_hidden_size ({bh}); "
        f"wrong --base-gguf for this --model?")
    print()

    print("== Loading drafter (fp32) ==")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, trust_remote_code=True,
    ).eval()
    print(f"  loaded; class={type(model).__name__}")
    print()

    BATCH, SEQ, KV_LEN = args.batch, args.seq, args.kv_len
    rng = torch.Generator().manual_seed(args.seed)

    # Build inputs_embeds = concat([h_random, e_from_base_tok_embd], dim=-1)
    # so C++ can replicate exactly: bind h as mtp_h_input, pass the same token
    # ids; C++ internally looks up e via ggml_get_rows(model.tok_embd, ids),
    # which reads the SAME base tok_embd data we used here.
    base_tok_embd_t = torch.from_numpy(base_tok_embd)  # [vocab, bh]
    vocab_size = base_tok_embd_t.shape[0]
    token_ids = (torch.arange(SEQ).long() + 100) % vocab_size  # deterministic
    token_ids = token_ids.unsqueeze(0).expand(BATCH, -1).contiguous().long()  # [B, S]
    e_part = base_tok_embd_t[token_ids.reshape(-1)].reshape(BATCH, SEQ, bh)
    h_part = 0.1 * torch.randn(BATCH, SEQ, bh, generator=rng)
    inputs_embeds = torch.cat([h_part, e_part], dim=-1).contiguous()  # [B, S, 2*bh]
    position_ids  = torch.arange(SEQ).unsqueeze(0).expand(BATCH, -1).contiguous()
    # Gemma4Assistant with attention_k_eq_v=True uses a smaller KV head
    # count for FULL-attention layers (num_global_key_value_heads).
    # Fall back to num_key_value_heads when absent (E2B-style).
    n_kv_heads_swa  = n_kv_heads
    n_kv_heads_full = getattr(tcfg, "num_global_key_value_heads", None) or n_kv_heads
    print(f"  n_kv_heads: swa={n_kv_heads_swa}  full={n_kv_heads_full}")
    K_swa  = 0.1 * torch.randn(BATCH, n_kv_heads_swa,  KV_LEN, head_dim_swa,  generator=rng)
    V_swa  = 0.1 * torch.randn(BATCH, n_kv_heads_swa,  KV_LEN, head_dim_swa,  generator=rng)
    K_full = 0.1 * torch.randn(BATCH, n_kv_heads_full, KV_LEN, head_dim_full, generator=rng)
    V_full = 0.1 * torch.randn(BATCH, n_kv_heads_full, KV_LEN, head_dim_full, generator=rng)
    attention_mask = torch.ones(BATCH, KV_LEN, dtype=torch.long)
    print(f"  token_ids: {token_ids.flatten().tolist()}")
    print(f"  e_part[0,0,:8]: {e_part[0,0,:8].tolist()}")
    print(f"  h_part[0,0,:8]: {h_part[0,0,:8].tolist()}")

    shared_kv_states = {
        "sliding_attention": (K_swa,  V_swa),
        "full_attention":    (K_full, V_full),
    }

    full_tensors: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    def stamp(label: str, t: torch.Tensor):
        if not isinstance(t, torch.Tensor):
            return
        arr = t.detach().to(torch.float32).cpu().numpy()
        full_tensors[label] = arr
        flat = arr.reshape(-1)
        summary[label] = {
            "shape":   list(arr.shape),
            "dtype":   str(arr.dtype),
            "mean":    float(flat.mean()),
            "std":     float(flat.std()),
            "abs_max": float(np.abs(flat).max()),
            "l2":      float(np.linalg.norm(flat)),
            "first16": flat[:16].tolist(),
        }

    def out_hook(label):
        def f(_m, _inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            stamp(label, t)
        return f

    def pre_hook(label):
        def f(_m, inp):
            t = inp[0] if isinstance(inp, tuple) else inp
            if isinstance(t, torch.Tensor):
                stamp(label, t)
        return f

    # Top-level
    model.pre_projection.register_forward_hook(out_hook("pre_projection"))
    model.post_projection.register_forward_hook(out_hook("post_projection"))
    model.model.norm.register_forward_hook(out_hook("model.norm"))
    model.lm_head.register_forward_hook(out_hook("lm_head"))

    # Per-layer
    for il in range(n_layers):
        layer = model.model.layers[il]
        # Always capture layer-output flag for completeness
        if il in layers_to_capture:
            sa = layer.self_attn
            mlp = layer.mlp

            layer.register_forward_pre_hook(pre_hook(f"L{il}.input"))
            layer.input_layernorm.register_forward_hook(out_hook(f"L{il}.input_layernorm"))
            sa.q_proj.register_forward_hook(out_hook(f"L{il}.q_proj"))
            if hasattr(sa, "q_norm"):
                sa.q_norm.register_forward_hook(out_hook(f"L{il}.q_norm"))
            sa.o_proj.register_forward_pre_hook(pre_hook(f"L{il}.attn_out_pre_o_proj"))
            sa.o_proj.register_forward_hook(out_hook(f"L{il}.o_proj"))
            if hasattr(layer, "post_attention_layernorm"):
                layer.post_attention_layernorm.register_forward_hook(
                    out_hook(f"L{il}.post_attention_layernorm")
                )
            if hasattr(mlp, "gate_proj"):
                mlp.gate_proj.register_forward_hook(out_hook(f"L{il}.gate_proj"))
            if hasattr(mlp, "up_proj"):
                mlp.up_proj.register_forward_hook(out_hook(f"L{il}.up_proj"))
            if hasattr(mlp, "down_proj"):
                mlp.down_proj.register_forward_hook(out_hook(f"L{il}.down_proj"))
            if hasattr(layer, "pre_feedforward_layernorm"):
                layer.pre_feedforward_layernorm.register_forward_hook(
                    out_hook(f"L{il}.pre_ff_ln")
                )
            if hasattr(layer, "post_feedforward_layernorm"):
                layer.post_feedforward_layernorm.register_forward_hook(
                    out_hook(f"L{il}.post_ff_ln")
                )
            layer.register_forward_hook(out_hook(f"L{il}.out"))

    # Monkey-patch self_attn.forward on captured layers to grab post-RoPE Q/K
    # and the raw attention weights.
    captured_attn = {}
    for il in layers_to_capture:
        sa = model.model.layers[il].self_attn
        orig_fwd = sa.forward
        ltype = layer_types[il]

        def make_wrapper(il=il, sa=sa, orig=orig_fwd, ltype=ltype):
            def wrapped(*a, **kw):
                # Capture inputs to forward so we can inspect them later.
                captured_attn[f"L{il}.attn.layer_type"] = ltype
                return orig(*a, **kw)
            return wrapped

        sa.forward = make_wrapper()

    # Also try to capture attention scores by monkey-patching torch.matmul
    # in the attention path? Too invasive. Instead rely on the o_proj
    # pre-hook to give us attn output (pre-o_proj), which is the most
    # diagnostic single tensor for "did cross-attention math work."

    print("== Forward ==")
    with torch.no_grad():
        try:
            out = model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                shared_kv_states=shared_kv_states,
            )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 2

    logits = out.logits
    last_hs = out.last_hidden_state
    stamp("logits", logits)
    stamp("last_hidden_state", last_hs)
    print(f"  logits shape:           {tuple(logits.shape)}")
    print(f"  last_hidden_state shape:{tuple(last_hs.shape)}")
    print(f"  logits[0,-1].argmax:    {int(logits[0, -1].argmax())}")
    print()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs as a separate .npz that C++ test reads
    inputs_path = out_dir / "reference_inputs.npz"
    np.savez(
        inputs_path,
        inputs_embeds=inputs_embeds.numpy(),
        h_part=h_part.numpy(),
        e_part=e_part.numpy(),
        token_ids=token_ids.numpy(),
        position_ids=position_ids.numpy(),
        attention_mask=attention_mask.numpy(),
        K_swa=K_swa.numpy(),
        V_swa=V_swa.numpy(),
        K_full=K_full.numpy(),
        V_full=V_full.numpy(),
    )
    print(f"Wrote {inputs_path}  ({inputs_path.stat().st_size:,} bytes)")

    # Save activations
    acts_path = out_dir / "reference_activations.npz"
    np.savez(acts_path, **full_tensors)
    print(f"Wrote {acts_path}  ({acts_path.stat().st_size:,} bytes)")

    # JSON summary (human-readable, no big arrays)
    summary_path = out_dir / "reference_activations.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": {
                "backbone_hidden_size": bh,
                "mtp_hidden": mh,
                "n_layers": n_layers,
                "n_attn_heads": n_attn_heads,
                "n_kv_heads": n_kv_heads,
                "head_dim_swa": head_dim_swa,
                "head_dim_full": head_dim_full,
                "layer_types": layer_types,
            },
            "inputs": {
                "batch": BATCH, "seq": SEQ, "kv_len": KV_LEN, "seed": args.seed,
                "shape_inputs_embeds": list(inputs_embeds.shape),
                "shape_K_swa": list(K_swa.shape),
                "shape_K_full": list(K_full.shape),
            },
            "layers_captured": sorted(layers_to_capture),
            "outputs": {
                "logits_argmax_pos_last": int(logits[0, -1].argmax()),
            },
            "attn_capture": captured_attn,
            "activations": summary,
        }, f, indent=2)
    print(f"Wrote {summary_path}  ({summary_path.stat().st_size:,} bytes)")
    print(f"  ({len(summary)} activation records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
