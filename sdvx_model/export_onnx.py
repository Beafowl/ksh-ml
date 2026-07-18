"""Export a trained checkpoint to a single KV-cached ONNX graph for
onnxruntime-web, and verify it numerically against torch.

  python -m sdvx_model.export_onnx --ckpt runs/audio/best.pt --out chartgen.onnx

Graph I/O (all batch/sequence dims dynamic):
  inputs : idx (B,S) int64, audio (B,S,A) f32, mask (B,1,S,P+S) f32 additive,
           past_k_i / past_v_i (B,nh,P,hd) f32 for each layer (P=0 allowed)
  outputs: logits (B,S,V) f32, pres_k_i / pres_v_i (B,nh,P+S,hd)
Also writes <out>.json with the model/tokenizer metadata the JS side needs.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from .model import ChartGPT, Config
from . import tokenizer as tok


class KVWrapper(nn.Module):
    def __init__(self, model: ChartGPT):
        super().__init__()
        self.m = model

    def forward(self, idx, audio, mask, *past_flat):
        past = [(past_flat[2 * i], past_flat[2 * i + 1])
                for i in range(len(past_flat) // 2)]
        logits, presents = self.m.forward_kv(idx, audio, mask, past)
        outs = [logits]
        for k, v in presents:
            outs += [k, v]
        return tuple(outs)


def causal_mask(s, p, batch=1):
    m = torch.zeros(batch, 1, s, p + s)
    if s > 1:
        tri = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
        m[:, :, :, p:] = tri
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="chartgen.onnx")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = Config(**ck["model_cfg"])
    model = ChartGPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    if ck["vocab"] != tok.VOCAB:
        raise SystemExit("checkpoint vocab differs from tokenizer")

    nl, nh, hd = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
    wrapper = KVWrapper(model)

    b, s, p = 1, 9, 0
    idx = torch.randint(4, cfg.vocab_size, (b, s))
    audio = torch.rand(b, s, cfg.audio_dim)
    mask = causal_mask(s, p, b)
    past = []
    for _ in range(nl):
        past += [torch.zeros(b, nh, p, hd), torch.zeros(b, nh, p, hd)]

    in_names = ["idx", "audio", "mask"]
    out_names = ["logits"]
    dyn = {"idx": {0: "B", 1: "S"}, "audio": {0: "B", 1: "S"},
           "mask": {0: "B", 2: "S", 3: "PS"}, "logits": {0: "B", 1: "S"}}
    for i in range(nl):
        in_names += [f"past_k_{i}", f"past_v_{i}"]
        out_names += [f"pres_k_{i}", f"pres_v_{i}"]
        dyn[f"past_k_{i}"] = dyn[f"past_v_{i}"] = {0: "B", 2: "P"}
        dyn[f"pres_k_{i}"] = dyn[f"pres_v_{i}"] = {0: "B", 2: "PS"}

    torch.onnx.export(
        wrapper, (idx, audio, mask, *past), args.out,
        input_names=in_names, output_names=out_names, dynamic_axes=dyn,
        opset_version=17, do_constant_folding=True)

    meta = {
        "vocab": tok.VOCAB,
        "model_cfg": cfg.__dict__,
        "n_layer": nl, "n_head": nh, "head_dim": hd,
        "prefix_len": tok.PREFIX_LEN,
        "radar_axes": tok.RADAR_AXES,
        "deltas": tok.DELTAS,
        "measure": tok.MEASURE,
        "grid": 12,
        "val_loss": ck.get("best_val"),
    }
    json.dump(meta, open(args.out + ".json", "w"), indent=1)

    # ---- parity check torch vs onnxruntime, prefill + one cached step ----
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])

    def run(idx_t, audio_t, mask_t, past_t):
        feeds = {"idx": idx_t.numpy(), "audio": audio_t.numpy(), "mask": mask_t.numpy()}
        for i in range(nl):
            feeds[f"past_k_{i}"] = past_t[2 * i].numpy()
            feeds[f"past_v_{i}"] = past_t[2 * i + 1].numpy()
        return sess.run(None, feeds)

    with torch.no_grad():
        ref = wrapper(idx, audio, mask, *past)
    out = run(idx, audio, mask, past)
    err = np.abs(ref[0].numpy() - out[0]).max()
    print(f"prefill parity: max |dlogits| = {err:.2e}")

    # one incremental step reusing the cache
    past2 = [torch.from_numpy(o) for o in out[1:]]
    idx2 = torch.randint(4, cfg.vocab_size, (b, 1))
    audio2 = torch.rand(b, 1, cfg.audio_dim)
    mask2 = causal_mask(1, s, b)
    with torch.no_grad():
        ref2 = wrapper(idx2, audio2, mask2,
                       *[t for t in past2])
    out2 = run(idx2, audio2, mask2, past2)
    err2 = np.abs(ref2[0].numpy() - out2[0]).max()
    print(f"step parity:    max |dlogits| = {err2:.2e}")
    assert err < 1e-3 and err2 < 1e-3, "ONNX export diverges from torch"
    import os
    print(f"exported {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) — OK")


if __name__ == "__main__":
    main()
