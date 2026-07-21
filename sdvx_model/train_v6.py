"""Train the v6 Whisper-style encoder-decoder on windowed chart+audio.

  python -m sdvx_model.train_v6 --charts out_v6/charts_v6.jsonl.gz \
      --mel out_v6/dense_mels.u8 --index out_v6/dense_index.json --out runs/v6

Each sample is a ~16 s audio window (1600 dense-mel frames) paired with the
chart events inside it, delta-encoded from the window start, behind the
conditioning prefix. Augmentation: lane mirror + tempo rate (audio frame
resample + bpm relabel) to stretch the 4.7k-chart corpus.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import time

import numpy as np
import torch

from .model_v6 import ChartV6, ConfigV6
from . import tokenizer_v6 as tk

FPS = 100.0
WINDOW_MS = 16000
WINDOW_FRAMES = 1600
CFG_DROP_ALL = 0.10
CFG_DROP_AXIS = 0.05
EFF_DROP = 0.30
MIRROR_P = 0.50
RATES = [0.85, 1.0, 1.0, 1.0, 1.0, 1.2, 1.5]  # tempo augmentation


def _mirror_perm() -> np.ndarray:
    """Column mirror: BT ABCD->DCBA, FX L<->R (radar labels are symmetric)."""
    perm = np.arange(len(tk.VOCAB), dtype=np.int64)
    for name, i in tk.ID.items():
        for kind in ("note", "hold_on", "hold_off"):
            if name.startswith(kind + "_"):
                c = int(name.rsplit("_", 1)[1])
                mc = (3 - c) if c < 4 else (4 + (5 - c))
                perm[i] = tk.ID[f"{kind}_{mc}"]
    return perm


MIRROR = _mirror_perm()


class WindowData:
    def __init__(self, charts, mel_path, index_path, val_mod=20):
        idx = json.load(open(index_path))
        self.fps = idx["fps"]
        total = idx["total_frames"]
        self.mel = np.memmap(mel_path, dtype=np.uint8, mode="r", shape=(total, idx["n_mels"]))
        self.frame_index = idx["index"]  # id -> [offset, n_frames]
        # effector vocab (top-100)
        from collections import Counter
        effc: Counter = Counter()
        with gzip.open(charts, "rt", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line).get("effected_by") or ""
                if e:
                    effc[e] += 1
        self.eff_names = [n for n, _ in effc.most_common(tk.EFF_VOCAB)]
        eff_map = {n: i for i, n in enumerate(self.eff_names)}

        self.train, self.val = [], []
        with gzip.open(charts, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                fi = self.frame_index.get(r["id"])
                if fi is None:
                    continue
                tokens, ticks = tk.encode_with_ticks(r, eff_map)
                bpm = (r.get("bpm") or [None, 120])[1] or r["chart"]["bpms"][0][1]
                item = {
                    "prefix": np.array(tokens[:tk.PREFIX_LEN], np.int64),
                    "body": np.array(tokens[tk.PREFIX_LEN:-1], np.int64),   # drop EOS
                    "bticks": np.array(ticks[tk.PREFIX_LEN:-1], np.int32),
                    "bpm": float(bpm),
                    "offset_ms": float(r.get("offset_ms") or 0),
                    "frame0": fi[0], "nframes": fi[1],
                }
                (self.val if (r["music_id"] or 0) % val_mod == 0 else self.train).append(item)
        print(f"  train {len(self.train)}, val {len(self.val)}, "
              f"effectors top-{len(self.eff_names)}")

    def _window(self, it, rng, train):
        bpm, off = it["bpm"], it["offset_ms"]
        r = rng.choice(RATES) if train else 1.0
        ms_per_tick = 60000.0 / bpm / 48.0
        win_ticks = int(WINDOW_MS * r / ms_per_tick)
        body, bticks = it["body"], it["bticks"]
        last_tick = int(bticks[-1]) if len(bticks) else 0
        if last_tick <= 0:
            start_tick = 0
        else:
            hi = max(1, last_tick - win_ticks // 2)
            start_tick = rng.randrange(0, hi) if train else 0
            start_tick -= start_tick % tk.MEASURE  # measure-align
        end_tick = start_tick + win_ticks
        # events in [start_tick, end_tick)
        lo = int(np.searchsorted(bticks, start_tick, "left"))
        hi = int(np.searchsorted(bticks, end_tick, "left"))
        seg_body = body[lo:hi]
        seg_ticks = bticks[lo:hi]
        # delta-encode from start_tick -> reuse encoder by rebuilding deltas
        toks = self._deltas(start_tick, seg_body, seg_ticks)

        # audio: original ms span [w0, w0 + WINDOW_MS*r] -> resample to 1600
        w0 = off + start_tick * ms_per_tick
        f0 = it["frame0"] + int(round(w0 / 1000.0 * self.fps))
        span = int(round(WINDOW_MS * r / 1000.0 * self.fps))  # original frames
        end_f = it["frame0"] + it["nframes"]
        src = np.zeros((max(span, 1), self.mel.shape[1]), np.float32)
        a = max(f0, it["frame0"]); b = min(f0 + span, end_f)
        if b > a:
            src[a - f0:b - f0] = self.mel[a:b].astype(np.float32) / 255.0
        # resample the frame axis to WINDOW_FRAMES (nearest)
        pick = np.minimum((np.arange(WINDOW_FRAMES) * span / WINDOW_FRAMES).astype(np.int64),
                          max(span - 1, 0))
        mel = src[pick]
        return it["prefix"].copy(), toks, mel, r

    def _deltas(self, start_tick, seg_body, seg_ticks):
        out = []
        pos = start_tick
        for tok, tick in zip(seg_body.tolist(), seg_ticks.tolist()):
            while pos < tick:
                nb = (pos // tk.MEASURE + 1) * tk.MEASURE
                if tick >= nb:
                    pos = nb; out.append(tk.ID["bar"])
                else:
                    d = max(n for n in tk.DELTAS if n <= tick - pos)
                    pos += d; out.append(tk.ID[f"d_{d}"])
            out.append(tok)
        return np.array(out, np.int64)

    def sample_batch(self, batch, rng, max_tgt, train=True):
        items = self.train if train else self.val
        xs, ys, mels = [], [], []
        for _ in range(batch):
            it = items[rng.randrange(len(items))]
            prefix, body, mel, r = self._window(it, rng, train)
            if train and rng.random() < MIRROR_P:
                body = MIRROR[body]
            # bpm relabel for rate aug
            if r != 1.0:
                prefix = prefix.copy()
                prefix[8] = tk.ID[f"bpm_{tk.bpm_bucket(it['bpm'] * r)}"]
            if train:
                if rng.random() < CFG_DROP_ALL:
                    prefix[1:8] = tk.UNCOND; prefix[tk.EFF_SLOT] = tk.UNCOND
                else:
                    for s in tk.RADAR_SLOTS:
                        if rng.random() < CFG_DROP_AXIS:
                            prefix[s] = tk.UNCOND
                    if rng.random() < EFF_DROP:
                        prefix[tk.EFF_SLOT] = tk.UNCOND
            seq = np.concatenate([prefix, body, [tk.EOS]])[:max_tgt + 1]
            x = seq[:-1]; y = seq[1:].copy()
            pad = max_tgt - len(x)
            if pad > 0:
                x = np.concatenate([x, np.full(pad, tk.PAD, np.int64)])
                y = np.concatenate([y, np.full(pad, -100, np.int64)])
            xs.append(x); ys.append(y); mels.append(mel)
        return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
                torch.from_numpy(np.stack(mels)).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", required=True)
    ap.add_argument("--mel", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--max-tgt", type=int, default=1024)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--enc-layer", type=int, default=6)
    ap.add_argument("--dec-layer", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device {device} bf16 {use_bf16}")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    data = WindowData(args.charts, args.mel, args.index)
    cfg = ConfigV6(vocab_size=len(tk.VOCAB), d_model=args.d_model,
                   enc_layer=args.enc_layer, dec_layer=args.dec_layer,
                   max_tgt=args.max_tgt, max_audio=WINDOW_FRAMES)
    model = ChartV6(cfg).to(device)
    print(f"  params {model.num_params()/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    start, best = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck.get("best_val", best)
        print(f"resumed at {start}")
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:  # noqa: BLE001
            print("compile off:", e)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        t = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.1 + 0.5 * args.lr * 0.9 * (1 + math.cos(math.pi * min(1, t)))

    def save(name, step):
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save({"model": raw.state_dict(), "opt": opt.state_dict(), "step": step,
                    "best_val": best, "model_cfg": cfg.__dict__, "vocab": tk.VOCAB,
                    "eff_names": data.eff_names}, os.path.join(args.out, name))

    @torch.no_grad()
    def evaluate():
        model.eval(); vr = random.Random(0); losses = []
        for _ in range(args.eval_batches):
            x, y, mel = data.sample_batch(args.batch, vr, args.max_tgt, train=False)
            x, y, mel = x.to(device), y.to(device), mel.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = model(mel, x, y)
            losses.append(loss.item())
        model.train(); return sum(losses) / len(losses)

    model.train(); t0, ntok = time.time(), 0
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True); total = 0.0
        for _ in range(args.accum):
            x, y, mel = data.sample_batch(args.batch, rng, args.max_tgt)
            x, y, mel = x.to(device), y.to(device), mel.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = model(mel, x, y)
            (loss / args.accum).backward()
            total += loss.item() / args.accum; ntok += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            dt = time.time() - t0
            print(f"step {step:5d} loss {total:.4f} lr {lr_at(step):.2e} "
                  f"{ntok/max(1e-9,dt)/1e3:.0f}k tok/s")
            t0, ntok = time.time(), 0
        if step > 0 and step % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(); flag = ""
            if vl < best:
                best = vl; save("best.pt", step); flag = "  (new best)"
            save("last.pt", step)
            print(f"eval @ {step}: val {vl:.4f}{flag}")
    print(f"done. best val {best:.4f}")


if __name__ == "__main__":
    main()
