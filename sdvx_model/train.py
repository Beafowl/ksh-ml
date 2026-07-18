"""Train ChartGPT on the dataset from sdvx_dataset.build.

  python -m sdvx_model.train --data out/charts.jsonl.gz --out runs/base

Defaults target an RTX 3070 (bf16 autocast, batch 16 x accum 2 x ctx 2048).
Checkpoints: <out>/last.pt every eval, <out>/best.pt on best val loss.
Resume with --resume <out>/last.pt.
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

from .model import ChartGPT, Config
from . import tokenizer as tok

CFG_DROP_ALL = 0.10    # replace level+radar with <uncond> (classifier-free guidance)
CFG_DROP_AXIS = 0.05   # additionally drop single radar axes (slider independence)
AUDIO_DROP = 0.10      # zero the onset features (keeps song-free generation working)
MIRROR_P = 0.50        # lane-mirror augmentation (radar labels are symmetric)
GRID = 12              # ticks per onset cell (matches the dataset's onset_step)
WINDOW = 16            # onset cells of lookahead per token


def _mirror_perm() -> np.ndarray:
    """Token permutation for the lane-mirrored chart: BT ABCD->DCBA, FX L<->R,
    lasers swap sides and flip values (v -> 50-v)."""
    perm = np.arange(len(tok.VOCAB), dtype=np.int64)
    for name, i in ((n, tok.ID[n]) for n in tok.VOCAB):
        m = None
        if name.startswith("bt_"):
            kind, lane = name.rsplit("_", 1)
            m = f"{kind}_{3 - int(lane)}"
        elif name.startswith("fx_"):
            kind, side = name.rsplit("_", 1)
            m = f"{kind}_{1 - int(side)}"
        elif name.startswith("la_v_"):
            _, _, side, v = name.split("_")
            m = f"la_v_{1 - int(side)}_{50 - int(v)}"
        elif name.startswith(("la_on_", "la_wide_", "la_off_")):
            kind, side = name.rsplit("_", 1)
            m = f"{kind}_{1 - int(side)}"
        if m is not None:
            perm[i] = tok.ID[m]
    return perm


MIRROR = _mirror_perm()


class ChartData:
    def __init__(self, path: str, ctx: int, audio_dim: int, val_mod: int = 20):
        self.ctx = ctx
        self.audio_dim = audio_dim
        self.train, self.val = [], []
        from collections import Counter
        lvl_count: Counter = Counter()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                tokens, ticks = tok.encode_with_ticks(r)
                onset = None
                if r.get("onset"):
                    arr = np.asarray(r["onset"], dtype=np.float32)
                    if arr.ndim == 1:  # v1 datasets: scalar onset per cell
                        arr = arr[:, None]
                    # pad with the lookahead window so indexing past the end is silent
                    onset = np.concatenate([arr, np.zeros((WINDOW, arr.shape[1]), dtype=np.float32)])
                    assert WINDOW * arr.shape[1] == audio_dim, (
                        f"--audio-dim must be {WINDOW * arr.shape[1]} for this dataset")
                item = {
                    "seq": np.array(tokens, dtype=np.int16),
                    "ticks": np.array(ticks, dtype=np.int32),
                    "onset": onset,
                    "level": int(r["level"]) or 1,
                }
                key = r["music_id"] if r["music_id"] is not None else hash(r["title"])
                if key % val_mod == 0:
                    self.val.append(item)
                else:
                    self.train.append(item)
                    lvl_count[item["level"]] += 1
        # length-weighted, flattened across levels so the easy end isn't drowned
        self.weights = np.array(
            [len(it["seq"]) / math.sqrt(lvl_count[it["level"]]) for it in self.train],
            dtype=np.float64)
        self.weights /= self.weights.sum()
        self.cum_weights = np.cumsum(self.weights)

    def _audio_feats(self, item, ticks: np.ndarray) -> np.ndarray:
        if item["onset"] is None:
            return np.zeros((len(ticks), self.audio_dim), dtype=np.float32)
        cells = (ticks // GRID)[:, None] + np.arange(WINDOW)[None, :]
        cells = np.minimum(cells, len(item["onset"]) - 1)
        return item["onset"][cells].reshape(len(ticks), -1)

    def sample_batch(self, batch: int, rng: random.Random, train=True):
        items = self.train if train else self.val
        xs, ys, aus = [], [], []
        for _ in range(batch):
            if train:
                i = int(np.searchsorted(self.cum_weights, rng.random()))
                i = min(i, len(items) - 1)
            else:
                i = rng.randrange(len(items))
            it = items[i]
            s = it["seq"].astype(np.int64)
            if train and rng.random() < MIRROR_P:
                s = MIRROR[s]
            tk = it["ticks"]
            prefix, body = s[:tok.PREFIX_LEN].copy(), s[tok.PREFIX_LEN:]
            tk_prefix, tk_body = tk[:tok.PREFIX_LEN], tk[tok.PREFIX_LEN:]
            if train:  # classifier-free guidance dropout
                if rng.random() < CFG_DROP_ALL:
                    prefix[1:1 + 7] = tok.UNCOND  # level + radar
                else:
                    for slot in tok.RADAR_SLOTS:
                        if rng.random() < CFG_DROP_AXIS:
                            prefix[slot] = tok.UNCOND
            body_ctx = self.ctx + 1 - len(prefix)
            if len(body) > body_ctx:
                start = rng.randrange(len(body) - body_ctx + 1) if train else 0
                body = body[start:start + body_ctx]
                tk_body = tk_body[start:start + body_ctx]
            x = np.concatenate([prefix, body])
            ticks_x = np.concatenate([tk_prefix, tk_body])
            pad = self.ctx + 1 - len(x)
            if pad > 0:
                x = np.concatenate([x, np.full(pad, tok.PAD, dtype=np.int64)])
                ticks_x = np.concatenate([ticks_x, np.full(pad, ticks_x[-1], dtype=np.int32)])
            au = self._audio_feats(it, ticks_x[:-1])
            if train and rng.random() < AUDIO_DROP:
                au = np.zeros_like(au)
            xs.append(x[:-1])
            y = x[1:].copy()
            y[y == tok.PAD] = -100
            ys.append(y)
            aus.append(au)
        return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
                torch.from_numpy(np.stack(aus)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--audio-dim", type=int, default=64, help="16 cells x features/cell; 0 disables audio")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device: {device} (bf16: {use_bf16})")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print("loading + tokenizing dataset ...")
    data = ChartData(args.data, args.ctx, args.audio_dim)
    n_tok = sum(len(it["seq"]) for it in data.train)
    print(f"  train charts: {len(data.train)} ({n_tok/1e6:.1f}M tokens), "
          f"val charts: {len(data.val)}, vocab: {len(tok.VOCAB)}")

    cfg = Config(vocab_size=len(tok.VOCAB), ctx=args.ctx, n_layer=args.n_layer,
                 n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
                 audio_dim=args.audio_dim)
    model = ChartGPT(cfg).to(device)
    print(f"  model params: {model.num_params()/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    start_step, best_val = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step, best_val = ck["step"], ck.get("best_val", best_val)
        print(f"resumed from {args.resume} at step {start_step}")

    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:  # noqa: BLE001 - Triton is often missing on Windows
            print("torch.compile unavailable:", e)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.1 + 0.5 * args.lr * 0.9 * (1 + math.cos(math.pi * min(1.0, t)))

    def save(name, step):
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                    "step": step, "best_val": best_val, "config": vars(args),
                    "model_cfg": cfg.__dict__, "vocab": tok.VOCAB},
                   os.path.join(args.out, name))

    @torch.no_grad()
    def evaluate():
        model.eval()
        vrng = random.Random(0)
        losses = []
        for _ in range(args.eval_batches):
            x, y, a = data.sample_batch(args.batch, vrng, train=False)
            x, y, a = x.to(device), y.to(device), a.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = model(x, y, audio=a)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    t0, tok_count = time.time(), 0
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accum):
            x, y, a = data.sample_batch(args.batch, rng)
            x, y, a = x.to(device), y.to(device), a.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = model(x, y, audio=a)
            (loss / args.accum).backward()
            total += loss.item() / args.accum
            tok_count += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0:
            dt = time.time() - t0
            print(f"step {step:5d}  loss {total:.4f}  lr {lr_at(step):.2e}  "
                  f"{tok_count/max(1e-9,dt)/1e3:.0f}k tok/s")
            t0, tok_count = time.time(), 0
        if step > 0 and step % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate()
            flag = ""
            if vl < best_val:
                best_val = vl
                save("best.pt", step)
                flag = "  (new best)"
            save("last.pt", step)
            print(f"eval @ {step}: val loss {vl:.4f}{flag}")
    print(f"done. best val loss {best_val:.4f}; checkpoints in {args.out}")


if __name__ == "__main__":
    main()
