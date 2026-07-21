"""Export a v6 checkpoint to two ONNX graphs for onnxruntime-web.

  encoder graph : mel (B,F,80) -> per-layer cross-attn K/V (run once/window)
  decoder graph : idx (B,S), self_mask (B,1,S,P+S), cross_k/v_i, self_past_k/v_i
                  -> logits (B,S,V), self_pres_k/v_i

  python -m sdvx_model.export_v6 --ckpt runs/v6/best.pt --out export/chartgen_v6.onnx [--fp16]

Verifies encoder output, decoder prefill, and one cached step against torch.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

from .model_v6 import ChartV6, ConfigV6
from . import tokenizer_v6 as tk


class EncWrapper(nn.Module):
    def __init__(self, m: ChartV6):
        super().__init__()
        self.m = m

    def forward(self, mel):
        memory = self.m.encode(mel)
        b, n = memory.shape[0], memory.shape[1]
        outs = []
        for blk in self.m.dec:
            outs.append(blk.cross_attn._split(blk.cross_attn.k(memory), b, n))
            outs.append(blk.cross_attn._split(blk.cross_attn.v(memory), b, n))
        return tuple(outs)


class DecWrapper(nn.Module):
    def __init__(self, m: ChartV6):
        super().__init__()
        self.m = m
        self.nl = m.cfg.dec_layer

    def forward(self, idx, self_mask, *rest):
        cross = [(rest[2 * i], rest[2 * i + 1]) for i in range(self.nl)]
        base = 2 * self.nl
        past = [(rest[base + 2 * i], rest[base + 2 * i + 1]) for i in range(self.nl)]
        pos = past[0][0].shape[2]
        logits, new_past = self.m.decode_step(idx, cross, past, pos, self_mask=self_mask)
        outs = [logits]
        for k, v in new_past:
            outs += [k, v]
        return tuple(outs)


def causal_mask(s, p, b=1):
    m = torch.zeros(b, 1, s, p + s)
    if s > 1:
        m[:, :, :, p:] = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="chartgen_v6.onnx")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ConfigV6(**ck["model_cfg"])
    model = ChartV6(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    if ck["vocab"] != tk.VOCAB:
        raise SystemExit("checkpoint vocab differs from tokenizer")
    nl, nh = cfg.dec_layer, cfg.n_head
    hd = cfg.d_model // nh
    enc_out = args.out[:-5] + ".enc.onnx" if args.out.endswith(".onnx") else args.out + ".enc.onnx"

    b, s, p, fr = 1, tk.PREFIX_LEN, 0, 300
    mel = torch.rand(b, fr, cfg.mel_bins)

    # ---- encoder ----
    enc_m, enc_in = EncWrapper(model), mel
    enc_names = []
    enc_dyn = {"mel": {0: "B", 1: "F"}}
    for i in range(nl):
        enc_names += [f"cross_k_{i}", f"cross_v_{i}"]
        enc_dyn[f"cross_k_{i}"] = enc_dyn[f"cross_v_{i}"] = {0: "B", 2: "N"}
    if args.fp16:
        enc_m = EncWrapper(copy.deepcopy(model).half()); enc_in = mel.half()
    torch.onnx.export(enc_m, (enc_in,), enc_out, input_names=["mel"],
                      output_names=enc_names, dynamic_axes=enc_dyn,
                      opset_version=17, do_constant_folding=True)

    # ---- decoder ----
    with torch.no_grad():
        memory = model.encode(mel)
    cross = []
    for blk in model.dec:
        cross += [blk.cross_attn._split(blk.cross_attn.k(memory), b, memory.shape[1]),
                  blk.cross_attn._split(blk.cross_attn.v(memory), b, memory.shape[1])]
    past = []
    for _ in range(nl):
        past += [torch.zeros(b, nh, p, hd), torch.zeros(b, nh, p, hd)]
    idx = torch.randint(4, cfg.vocab_size, (b, s))
    mask = causal_mask(s, p, b)

    dec_in_names = ["idx", "self_mask"]
    dec_out_names = ["logits"]
    dec_dyn = {"idx": {0: "B", 1: "S"}, "self_mask": {0: "B", 2: "S", 3: "PS"},
               "logits": {0: "B", 1: "S"}}
    for i in range(nl):
        dec_in_names += [f"cross_k_{i}", f"cross_v_{i}"]
        dec_dyn[f"cross_k_{i}"] = dec_dyn[f"cross_v_{i}"] = {0: "B", 2: "N"}
    for i in range(nl):
        dec_in_names += [f"past_k_{i}", f"past_v_{i}"]
        dec_out_names += [f"pres_k_{i}", f"pres_v_{i}"]
        dec_dyn[f"past_k_{i}"] = dec_dyn[f"past_v_{i}"] = {0: "B", 2: "P"}
        dec_dyn[f"pres_k_{i}"] = dec_dyn[f"pres_v_{i}"] = {0: "B", 2: "PS"}

    dec_m = DecWrapper(model)
    dec_inputs = (idx, mask, *[t for pair in
                  [(cross[2 * i], cross[2 * i + 1]) for i in range(nl)] for t in pair], *past)
    if args.fp16:
        dec_m = DecWrapper(copy.deepcopy(model).half())
        dec_inputs = (idx, mask.half(),
                      *[t.half() for t in [cross[j] for j in range(2 * nl)]],
                      *[t.half() for t in past])
    torch.onnx.export(dec_m, dec_inputs, args.out, input_names=dec_in_names,
                      output_names=dec_out_names, dynamic_axes=dec_dyn,
                      opset_version=17, do_constant_folding=True)

    meta = {"vocab": tk.VOCAB, "fp16": bool(args.fp16), "model_cfg": cfg.__dict__,
            "n_layer": nl, "n_head": nh, "head_dim": hd, "d_model": cfg.d_model,
            "prefix_len": tk.PREFIX_LEN, "eff_names": ck.get("eff_names") or [],
            "radar_axes": tk.RADAR_AXES, "deltas": tk.DELTAS, "measure": tk.MEASURE,
            "mel_bins": cfg.mel_bins, "window_frames": cfg.max_audio, "fps": 100,
            "val_loss": ck.get("best_val")}
    json.dump(meta, open(args.out + ".json", "w"), indent=1)

    # ---- parity ----
    import onnxruntime as ort
    ft = np.float16 if args.fp16 else np.float32
    es = ort.InferenceSession(enc_out, providers=["CPUExecutionProvider"])
    ecross = es.run(None, {"mel": mel.numpy().astype(ft)})
    ref_cross = [t.detach().numpy() for t in cross]
    err_e = max(np.abs(ref_cross[i] - ecross[i].astype(np.float32)).max() for i in range(2 * nl))
    print(f"encoder cross-KV parity: max |d| = {err_e:.2e}")

    ds = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])

    def run(idx_t, mask_t, past_t):
        feeds = {"idx": idx_t.numpy(), "self_mask": mask_t.numpy().astype(ft)}
        for i in range(nl):
            feeds[f"cross_k_{i}"] = ecross[2 * i].astype(ft)
            feeds[f"cross_v_{i}"] = ecross[2 * i + 1].astype(ft)
        for i in range(nl):
            feeds[f"past_k_{i}"] = past_t[2 * i].numpy().astype(ft)
            feeds[f"past_v_{i}"] = past_t[2 * i + 1].numpy().astype(ft)
        return ds.run(None, feeds)

    with torch.no_grad():
        ref = dec_m.float()(idx, mask, *[cross[j] for j in range(2 * nl)], *past) \
            if not args.fp16 else DecWrapper(model)(idx, mask, *[cross[j] for j in range(2 * nl)], *past)
    out = run(idx, mask, past)
    err = np.abs(ref[0].numpy() - out[0].astype(np.float32)).max()
    print(f"decoder prefill parity:  max |dlogits| = {err:.2e}")

    past2 = [torch.from_numpy(o.astype(np.float32)) for o in out[1:]]
    idx2 = torch.randint(4, cfg.vocab_size, (b, 1))
    mask2 = causal_mask(1, s, b)
    with torch.no_grad():
        ref2 = DecWrapper(model)(idx2, mask2, *[cross[j] for j in range(2 * nl)], *past2)
    out2 = run(idx2, mask2, past2)
    err2 = np.abs(ref2[0].numpy() - out2[0].astype(np.float32)).max()
    print(f"decoder step parity:     max |dlogits| = {err2:.2e}")
    tol = 2e-1 if args.fp16 else 1e-3
    assert err_e < tol and err < tol and err2 < tol, "v6 ONNX export diverges"
    print(f"exported {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB) + "
          f"{enc_out} ({os.path.getsize(enc_out)/1e6:.1f} MB) — OK")


if __name__ == "__main__":
    main()
