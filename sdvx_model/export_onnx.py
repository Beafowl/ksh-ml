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


class KVWrapperV5(nn.Module):
    """v5 decoder: audio comes as encoder memory + per-token cell indices."""

    def __init__(self, model: ChartGPT):
        super().__init__()
        self.m = model

    def forward(self, idx, memory, cells, mask, *past_flat):
        past = [(past_flat[2 * i], past_flat[2 * i + 1])
                for i in range(len(past_flat) // 2)]
        logits, presents = self.m.forward_kv(idx, None, mask, past,
                                             memory=memory, cells=cells)
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


def export_v5(model: ChartGPT, cfg: Config, ck: dict, args):
    """Two-graph export: encoder (mel -> memory, run once per song) and a
    KV-cached decoder that reads memory via per-token cell indices."""
    import copy
    import os

    import onnxruntime as ort

    nl, nh, hd = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
    enc_out = args.out[:-5] + ".enc.onnx" if args.out.endswith(".onnx") \
        else args.out + ".enc.onnx"
    b, s, p, n = 1, tok.PREFIX_LEN, 0, 200
    mel = torch.rand(b, n, cfg.mel_bins)

    # ---- encoder graph ----
    enc_m, enc_in = model.encoder, mel
    if args.fp16:
        enc_m, enc_in = copy.deepcopy(model.encoder).half(), mel.half()
    torch.onnx.export(
        enc_m, (enc_in,), enc_out, input_names=["mel"], output_names=["memory"],
        dynamic_axes={"mel": {0: "B", 1: "N"}, "memory": {0: "B", 1: "N"}},
        opset_version=17, do_constant_folding=True)

    # ---- decoder graph ----
    wrapper = KVWrapperV5(model)
    idx = torch.randint(4, cfg.vocab_size, (b, s))
    with torch.no_grad():
        memory = model.encoder(mel)
    cells = torch.randint(0, n, (b, s))
    mask = causal_mask(s, p, b)
    past = []
    for _ in range(nl):
        past += [torch.zeros(b, nh, p, hd), torch.zeros(b, nh, p, hd)]

    in_names = ["idx", "memory", "cells", "mask"]
    out_names = ["logits"]
    dyn = {"idx": {0: "B", 1: "S"}, "memory": {0: "B", 1: "N"},
           "cells": {0: "B", 1: "S"}, "mask": {0: "B", 2: "S", 3: "PS"},
           "logits": {0: "B", 1: "S"}}
    for i in range(nl):
        in_names += [f"past_k_{i}", f"past_v_{i}"]
        out_names += [f"pres_k_{i}", f"pres_v_{i}"]
        dyn[f"past_k_{i}"] = dyn[f"past_v_{i}"] = {0: "B", 2: "P"}
        dyn[f"pres_k_{i}"] = dyn[f"pres_v_{i}"] = {0: "B", 2: "PS"}

    export_wrapper = wrapper
    export_inputs = (idx, memory, cells, mask, *past)
    if args.fp16:
        m16 = copy.deepcopy(model).half()
        export_wrapper = KVWrapperV5(m16)
        export_inputs = (idx, memory.half(), cells, mask.half(),
                         *[t.half() for t in past])
    torch.onnx.export(
        export_wrapper, export_inputs, args.out,
        input_names=in_names, output_names=out_names, dynamic_axes=dyn,
        opset_version=17, do_constant_folding=True)

    meta = {
        "vocab": tok.VOCAB,
        "fp16": bool(args.fp16),
        "model_cfg": cfg.__dict__,
        "n_layer": nl, "n_head": nh, "head_dim": hd,
        "enc_embd": cfg.enc_embd, "max_cells": cfg.max_cells,
        "prefix_len": tok.PREFIX_LEN,
        "eff_names": ck.get("eff_names") or [],
        "radar_axes": tok.RADAR_AXES,
        "deltas": tok.DELTAS,
        "measure": tok.MEASURE,
        "grid": 12,
        "val_loss": ck.get("best_val"),
    }
    json.dump(meta, open(args.out + ".json", "w"), indent=1)

    # ---- parity: encoder, then decoder prefill + one cached step ----
    ftype = np.float16 if args.fp16 else np.float32
    es = ort.InferenceSession(enc_out, providers=["CPUExecutionProvider"])
    mem_ort = es.run(None, {"mel": mel.numpy().astype(ftype)})[0]
    err_e = np.abs(memory.numpy() - mem_ort.astype(np.float32)).max()
    print(f"encoder parity: max |dmem| = {err_e:.2e}")

    ds = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])

    def run(idx_t, cells_t, mask_t, past_t):
        feeds = {"idx": idx_t.numpy(), "memory": mem_ort.astype(ftype),
                 "cells": cells_t.numpy(), "mask": mask_t.numpy().astype(ftype)}
        for i in range(nl):
            feeds[f"past_k_{i}"] = past_t[2 * i].numpy().astype(ftype)
            feeds[f"past_v_{i}"] = past_t[2 * i + 1].numpy().astype(ftype)
        return ds.run(None, feeds)

    with torch.no_grad():
        ref = wrapper(idx, memory, cells, mask, *past)
    out = run(idx, cells, mask, past)
    err = np.abs(ref[0].numpy() - out[0].astype(np.float32)).max()
    print(f"prefill parity: max |dlogits| = {err:.2e}")

    past2 = [torch.from_numpy(o.astype(np.float32)) for o in out[1:]]
    idx2 = torch.randint(4, cfg.vocab_size, (b, 1))
    cells2 = torch.randint(0, n, (b, 1))
    mask2 = causal_mask(1, s, b)
    with torch.no_grad():
        ref2 = wrapper(idx2, memory, cells2, mask2, *past2)
    out2 = run(idx2, cells2, mask2, past2)
    err2 = np.abs(ref2[0].numpy() - out2[0].astype(np.float32)).max()
    print(f"step parity:    max |dlogits| = {err2:.2e}")
    tol = 1e-1 if args.fp16 else 1e-3
    assert err_e < tol and err < tol and err2 < tol, "ONNX export diverges"
    print(f"exported {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) + "
          f"{enc_out} ({os.path.getsize(enc_out) / 1e6:.1f} MB) — OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="chartgen.onnx")
    ap.add_argument("--fp16", action="store_true",
                    help="full fp16 graph (float I/O becomes fp16; halves the file)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = Config(**ck["model_cfg"])
    model = ChartGPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    if ck["vocab"] != tok.VOCAB:
        raise SystemExit("checkpoint vocab differs from tokenizer")

    if cfg.enc_layer:
        export_v5(model, cfg, ck, args)
        return

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

    # fp16: export natively from a half-precision copy (post-hoc converters
    # mishandle the KV-cache pass-through outputs)
    export_wrapper, export_inputs = wrapper, (idx, audio, mask, *past)
    if args.fp16:
        import copy
        m16 = copy.deepcopy(model).half()
        export_wrapper = KVWrapper(m16)
        export_inputs = (idx, audio.half(), mask.half(), *[t.half() for t in past])
    torch.onnx.export(
        export_wrapper, export_inputs, args.out,
        input_names=in_names, output_names=out_names, dynamic_axes=dyn,
        opset_version=17, do_constant_folding=True)

    meta = {
        "vocab": tok.VOCAB,
        "fp16": bool(args.fp16),
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

    ftype = np.float16 if args.fp16 else np.float32

    def run(idx_t, audio_t, mask_t, past_t):
        feeds = {"idx": idx_t.numpy(),
                 "audio": audio_t.numpy().astype(ftype),
                 "mask": mask_t.numpy().astype(ftype)}
        for i in range(nl):
            feeds[f"past_k_{i}"] = past_t[2 * i].numpy().astype(ftype)
            feeds[f"past_v_{i}"] = past_t[2 * i + 1].numpy().astype(ftype)
        return sess.run(None, feeds)

    with torch.no_grad():
        ref = wrapper(idx, audio, mask, *past)
    out = run(idx, audio, mask, past)
    err = np.abs(ref[0].numpy() - out[0].astype(np.float32)).max()
    print(f"prefill parity: max |dlogits| = {err:.2e}")

    # one incremental step reusing the cache
    past2 = [torch.from_numpy(o.astype(np.float32)) for o in out[1:]]
    idx2 = torch.randint(4, cfg.vocab_size, (b, 1))
    audio2 = torch.rand(b, 1, cfg.audio_dim)
    mask2 = causal_mask(1, s, b)
    with torch.no_grad():
        ref2 = wrapper(idx2, audio2, mask2,
                       *[t for t in past2])
    out2 = run(idx2, audio2, mask2, past2)
    err2 = np.abs(ref2[0].numpy() - out2[0].astype(np.float32)).max()
    print(f"step parity:    max |dlogits| = {err2:.2e}")
    tol = 1e-1 if args.fp16 else 1e-3  # fp16 rounding is fine at logit scale
    assert err < tol and err2 < tol, "ONNX export diverges from torch"
    import os
    print(f"exported {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) — OK")


if __name__ == "__main__":
    main()
