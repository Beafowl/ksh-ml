"""Generate a chart from a trained checkpoint and write it as .ksh.

  python -m sdvx_model.sample --ckpt runs/base/best.pt --out gen.ksh \
      --level 17 --notes 0.7 --peak 0.6 --tsumami 0.4 --tricky 0.1 \
      --hand-trip 0.3 --one-hand 0.4 --bpm 180 --measures 64 --guidance 2.0

Sliders are 0..1 (mapped onto the official 0-200 radar scale). Guidance > 1
pushes generation toward the sliders (1.0 disables classifier-free guidance).
Open the result in the editor (drag & drop the .ksh) to inspect and clean up.
"""
from __future__ import annotations

import argparse

import torch

from .model import ChartGPT, Config
from . import tokenizer as tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="gen.ksh")
    ap.add_argument("--level", type=int, default=15)
    for ax in tok.RADAR_AXES:
        ap.add_argument(f"--{ax}", type=float, default=0.5, help="0..1")
    ap.add_argument("--bpm", type=float, default=170.0)
    ap.add_argument("--measures", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location=device)
    if ck["vocab"] != tok.VOCAB:
        raise SystemExit("checkpoint vocab differs from current tokenizer — "
                         "sample with the code version that trained it")
    model = ChartGPT(Config(**ck["model_cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    radar = {ax: round(max(0.0, min(1.0, getattr(args, ax.replace("-", "_")))) * 200)
             for ax in tok.RADAR_AXES}
    cond = [tok.BOS] + tok.cond_tokens(args.level, radar, args.bpm)
    uncond = [tok.BOS, cond[1]] + [tok.UNCOND] * 6 + [cond[8]]
    use_cfg = args.guidance is not None and abs(args.guidance - 1.0) > 1e-6

    body: list[int] = []
    pos = 0
    print(f"generating (level {args.level}, radar {radar}, bpm {args.bpm:g}, "
          f"guidance {args.guidance}) ...")
    with torch.no_grad():
        while len(body) < args.max_tokens and pos < args.measures * tok.MEASURE:
            rows = [cond + body, uncond + body] if use_cfg else [cond + body]
            width = max(len(r) for r in rows)
            idx = torch.tensor([r[-model.cfg.ctx:] for r in rows], device=device)
            t = model.generate_step(idx, args.temperature, args.top_p,
                                    args.guidance if use_cfg else None)
            if t == tok.EOS:
                break
            if t in (tok.PAD, tok.BOS):
                continue
            body.append(t)
            name = tok.VOCAB[t]
            if name == "bar":
                pos = (pos // tok.MEASURE + 1) * tok.MEASURE
            elif name.startswith("d_"):
                pos += int(name[2:])
            if len(body) % 500 == 0:
                print(f"  {len(body)} tokens, measure {pos // tok.MEASURE}")

    chart = tok.decode_body(body)
    n_notes = sum(len(l) for l in chart["bt"]) + sum(len(s) for s in chart["fx"])
    n_pts = sum(len(g['pts']) for s in chart["lasers"] for g in s)
    print(f"decoded: {n_notes} notes, {n_pts} laser points, "
          f"{pos // tok.MEASURE} measures")
    text = tok.chart_to_ksh(
        chart, title=f"generated lv{args.level}", level=args.level, bpm=args.bpm)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
