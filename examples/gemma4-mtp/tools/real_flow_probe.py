#!/usr/bin/env python3
"""Real-flow HF reference: run base + drafter on a REAL prompt, capture h_t,
shared K/V, drafter activations and logits. Used to verify our C++ port's
base-capture path against HF's authoritative flow.

Outputs:
  /tmp/real_inputs.npz       - tokens, prompt, h_t (post-norm from base's last
                               token), shared_kv_states dict
  /tmp/real_acts.npz         - drafter activations + logits keyed like
                               reference_activations.npz
  /tmp/real_inputs.json/.bin - flat-file versions for C++ harness
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


DEFAULT_BASE  = "google/gemma-4-E2B-it"
DEFAULT_DRAFT = "google/gemma-4-E2B-it-assistant"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base",   default=DEFAULT_BASE)
    p.add_argument("--draft",  default=DEFAULT_DRAFT)
    p.add_argument("--prompt", default="The quick brown fox")
    p.add_argument("--out",    default="/tmp")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)

    print("== Loading base + drafter (fp32) ==")
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.float32, trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, dtype=torch.float32, trust_remote_code=True).eval()
    bh = base.config.text_config.hidden_size
    print(f"  base hidden_size={bh}")

    # Match our C++ test which uses add_special=true (BOS prepended).
    # HF tokenizer doesn't add BOS by default for raw text, so prepend manually.
    inputs = tok(args.prompt, return_tensors="pt", add_special_tokens=False)
    bos_id = tok.bos_token_id or 2
    ids = torch.cat([torch.tensor([[bos_id]]), inputs.input_ids], dim=1)
    print(f"  prompt: {args.prompt!r}  (BOS={bos_id} prepended)")
    print(f"  token_ids: {ids.flatten().tolist()}")

    # Run base with return_shared_kv_states + output_hidden_states so we can
    # extract h (last_hidden_state at last position) and shared K/V tuples.
    # The TEXT model lives at base.model.language_model in Gemma4 multimodal.
    text_model = base.model.language_model
    print("\n== Base forward (capture h + shared K/V) ==")
    with torch.no_grad():
        out = text_model(
            input_ids=ids,
            return_shared_kv_states=True,
            output_hidden_states=True,
        )
    base_logits = base.lm_head(out.last_hidden_state)
    base_argmax = int(base_logits[0, -1].argmax())
    print(f"  base predicts next token: {base_argmax} → {tok.decode([base_argmax])!r}")
    h_t = out.last_hidden_state[0, -1].detach().cpu().numpy().astype(np.float32)
    print(f"  h_t (last_hidden_state at last pos): shape={h_t.shape}")

    shared_kv = out.shared_kv_states
    print(f"  shared_kv_states keys: {list(shared_kv.keys())}")
    K_swa, V_swa   = shared_kv["sliding_attention"]
    K_full, V_full = shared_kv["full_attention"]
    print(f"  K_swa  shape={tuple(K_swa.shape)}  dtype={K_swa.dtype}")
    print(f"  K_full shape={tuple(K_full.shape)}  dtype={K_full.dtype}")

    K_swa_np  = K_swa.detach().cpu().numpy().astype(np.float32)
    V_swa_np  = V_swa.detach().cpu().numpy().astype(np.float32)
    K_full_np = K_full.detach().cpu().numpy().astype(np.float32)
    V_full_np = V_full.detach().cpu().numpy().astype(np.float32)

    # Build inputs_embeds = concat(h_t, embed(base_argmax)) for the drafter.
    # The drafter does skip-1 prediction: given (h_t, e_{t+1}) predict t+2.
    e_next = text_model.embed_tokens.weight[base_argmax].detach().cpu().numpy().astype(np.float32)
    # Note: base.model.embed_tokens is a ScaledWordEmbedding (multiplies by
    # sqrt(hidden) at lookup). The .weight is the raw value. The drafter
    # gets the SCALED embedding when called via forward; but when we
    # construct inputs_embeds externally, we should NOT apply the scale
    # (it's already applied via the lookup in HF's normal flow). Look at
    # the assistant's forward: it accepts inputs_embeds directly without
    # rescaling. We're following the convention used in reference_probe.py:
    # use raw (unscaled) base tok_embd row as e_part. (Same as our C++
    # which uses model.tok_embd via ggml_get_rows, unscaled.)
    print(f"  e_next (raw embedding of base_argmax={base_argmax}): shape={e_next.shape}")

    inputs_embeds = np.concatenate([h_t, e_next])  # [2*bh]
    inputs_embeds = inputs_embeds.reshape(1, 1, -1)
    print(f"  inputs_embeds shape={inputs_embeds.shape}")

    # Drafter forward
    print("\n== Drafter forward ==")
    # Hook activations
    acts = {}
    def stamp(name, t):
        if not isinstance(t, torch.Tensor): return
        arr = t.detach().to(torch.float32).cpu().numpy()
        acts[name] = arr
    def out_hook(name):
        def f(_m, _i, o): stamp(name, o if isinstance(o, torch.Tensor) else o[0])
        return f
    def pre_hook(name):
        def f(_m, i): stamp(name, i[0] if isinstance(i, tuple) else i)
        return f
    draft.pre_projection.register_forward_hook(out_hook("pre_projection"))
    draft.post_projection.register_forward_hook(out_hook("post_projection"))
    draft.model.norm.register_forward_hook(out_hook("model.norm"))
    L0 = draft.model.layers[0]
    L0.input_layernorm.register_forward_hook(out_hook("L0.input_layernorm"))
    L0.self_attn.q_proj.register_forward_hook(out_hook("L0.q_proj"))
    L0.self_attn.q_norm.register_forward_hook(out_hook("L0.q_norm"))
    L0.self_attn.o_proj.register_forward_pre_hook(pre_hook("L0.attn_out_pre_o_proj"))
    L0.self_attn.o_proj.register_forward_hook(out_hook("L0.o_proj"))
    L0.register_forward_hook(out_hook("L0.out"))

    pos_ids = torch.tensor([[ids.shape[1] - 1]], dtype=torch.long)  # position of t (last prompt token)
    attn_mask = torch.ones(1, K_swa.shape[2], dtype=torch.long)

    with torch.no_grad():
        d_out = draft(
            inputs_embeds=torch.from_numpy(inputs_embeds),
            position_ids=pos_ids,
            attention_mask=attn_mask,
            shared_kv_states={
                "sliding_attention": (K_swa, V_swa),
                "full_attention":    (K_full, V_full),
            },
        )

    logits = d_out.logits.detach().cpu().numpy().astype(np.float32)
    last_hs = d_out.last_hidden_state.detach().cpu().numpy().astype(np.float32)
    draft_argmax = int(logits.flatten().argmax())
    print(f"  drafter argmax = {draft_argmax} → {tok.decode([draft_argmax])!r}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs (h_t, K/V, token_ids, etc.) for C++ harness replication
    np.savez(out_dir / "real_inputs.npz",
             token_ids=ids.numpy().astype(np.int32),
             base_argmax=np.array([base_argmax], dtype=np.int32),
             h_t=h_t,
             e_next=e_next,
             inputs_embeds=inputs_embeds.astype(np.float32),
             K_swa=K_swa_np, V_swa=V_swa_np,
             K_full=K_full_np, V_full=V_full_np,
             attention_mask=attn_mask.numpy(),
             position_ids=pos_ids.numpy())
    np.savez(out_dir / "real_acts.npz", logits=logits, last_hidden_state=last_hs, **acts)
    print(f"\nWrote {out_dir}/real_inputs.npz and real_acts.npz")
    print(f"  expected drafter argmax in real flow: {draft_argmax}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
