"""Generate a chart from a trained checkpoint and write it as .ksh.

  python -m sdvx_model.sample --ckpt runs/audio/best.pt --out gen.ksh \
      --level 17 --notes 0.7 --peak 0.6 --tsumami 0.4 --tricky 0.1 \
      --hand-trip 0.3 --one-hand 0.4 --bpm 180 --measures 64 --guidance 2.0 \
      --audio song.ogg --offset 40

Sliders are 0..1 (mapped onto the official 0-200 radar scale). Guidance > 1
pushes generation toward the sliders (1.0 disables classifier-free guidance).
With --audio, note placement follows the song's onsets (assumes constant
--bpm; --offset = ms where beat 1 lands, like the ksh 'o' field). Without
--audio, the model generates pattern-only, as in training's audio dropout.
Open the result in the editor (drag & drop the .ksh) to inspect and clean up.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .model import ChartGPT, Config
from . import tokenizer as tok

GRID = 12  # ticks per onset cell


def song_onsets(path: str, bpm: float, offset_ms: float, n_cells: int, audio_dim: int):
    from sdvx_dataset.onsets import grid_onsets, onset_envelope
    env, fps = onset_envelope(path)
    grid_ms = offset_ms + np.arange(n_cells, dtype=np.float64) * (15000.0 / bpm)
    vals = grid_onsets(env, fps, grid_ms).astype(np.float32)
    return np.concatenate([vals, np.zeros(audio_dim, dtype=np.float32)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="gen.ksh")
    ap.add_argument("--level", type=int, default=15)
    for ax in tok.RADAR_AXES:
        ap.add_argument(f"--{ax}", type=float, default=0.5, help="0..1")
    ap.add_argument("--bpm", type=float, default=170.0)
    ap.add_argument("--measures", type=int, default=64)
    ap.add_argument("--audio", default=None, help="song file to follow (ogg)")
    ap.add_argument("--offset", type=float, default=0.0, help="ms of beat 1 in the audio")
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
    audio_dim = model.cfg.audio_dim

    n_cells = args.measures * tok.MEASURE // GRID + 1
    if args.audio and audio_dim:
        onsets = song_onsets(args.audio, args.bpm, args.offset, n_cells, audio_dim)
        print(f"audio: {args.audio} -> {n_cells} onset cells")
    else:
        onsets = np.zeros(n_cells + audio_dim, dtype=np.float32)

    def window(pos: int) -> np.ndarray:
        c = min(pos // GRID, len(onsets) - audio_dim - 1)
        return onsets[c:c + audio_dim]

    radar = {ax: round(max(0.0, min(1.0, getattr(args, ax.replace("-", "_")))) * 200)
             for ax in tok.RADAR_AXES}
    cond = [tok.BOS] + tok.cond_tokens(args.level, radar, args.bpm)
    uncond = [tok.BOS, cond[1]] + [tok.UNCOND] * 6 + [cond[8]]
    use_cfg = args.guidance is not None and abs(args.guidance - 1.0) > 1e-6

    body: list[int] = []
    feats: list[np.ndarray] = [window(0)] * tok.PREFIX_LEN  # prefix rows
    pos = 0
    print(f"generating (level {args.level}, radar {radar}, bpm {args.bpm:g}, "
          f"guidance {args.guidance}, audio={'yes' if args.audio else 'no'}) ...")
    with torch.no_grad():
        while len(body) < args.max_tokens and pos < args.measures * tok.MEASURE:
            rows = [cond + body, uncond + body] if use_cfg else [cond + body]
            idx = torch.tensor([r[-model.cfg.ctx:] for r in rows], device=device)
            au = None
            if audio_dim:
                a = np.stack(feats[-model.cfg.ctx:])
                au = torch.from_numpy(np.stack([a] * len(rows))).to(device)
            t = model.generate_step(idx, args.temperature, args.top_p,
                                    args.guidance if use_cfg else None, audio=au)
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
            feats.append(window(pos))
            if len(body) % 500 == 0:
                print(f"  {len(body)} tokens, measure {pos // tok.MEASURE}")

    chart = tok.decode_body(body)
    n_notes = sum(len(l) for l in chart["bt"]) + sum(len(s) for s in chart["fx"])
    n_pts = sum(len(g['pts']) for s in chart["lasers"] for g in s)
    print(f"decoded: {n_notes} notes, {n_pts} laser points, "
          f"{pos // tok.MEASURE} measures")
    import os
    music = os.path.basename(args.audio) if args.audio else ""
    text = tok.chart_to_ksh(
        chart, title=f"generated lv{args.level}", level=args.level, bpm=args.bpm,
        music_file=music, offset_ms=args.offset)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
