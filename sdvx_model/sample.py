"""Generate a chart from a trained checkpoint and write it as .ksh.

  python -m sdvx_model.sample --ckpt runs/audio2/best.pt --out gen.ksh \
      --level 17 --notes 0.7 --peak 0.6 --tsumami 0.4 --tricky 0.1 \
      --hand-trip 0.3 --one-hand 0.4 --bpm 180 --measures 64 \
      --guidance 2.0 --audio-guidance 2.5 --audio song.ogg --offset 40

Sliders are 0..1 (mapped onto the official 0-200 radar scale).
--guidance steers toward the sliders; --audio-guidance amplifies how strongly
generation follows the song (both 1.0 = off). Without --audio the model
generates pattern-only.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .model import ChartGPT, Config
from . import tokenizer as tok

GRID = 12
WINDOW = 16


def song_features(path: str, bpm: float, offset_ms: float, n_cells: int, feats_per_cell: int):
    from sdvx_dataset.onsets import grid_features, grid_onsets, onset_envelope, onset_features
    grid_ms = offset_ms + np.arange(n_cells, dtype=np.float64) * (15000.0 / bpm)
    if feats_per_cell == 1:
        env, fps = onset_envelope(path)
        vals = grid_onsets(env, fps, grid_ms).astype(np.float32)[:, None]
    else:
        feats, fps = onset_features(path)
        vals = grid_features(feats, fps, grid_ms).astype(np.float32)
    return np.concatenate([vals, np.zeros((WINDOW, vals.shape[1]), dtype=np.float32)])


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
    ap.add_argument("--audio-guidance", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location=device)
    if ck["vocab"] != tok.VOCAB:
        raise SystemExit("checkpoint vocab differs from current tokenizer")
    model = ChartGPT(Config(**ck["model_cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    audio_dim = model.cfg.audio_dim
    feats_per_cell = max(1, audio_dim // WINDOW)

    n_cells = args.measures * tok.MEASURE // GRID + 1
    have_audio = bool(args.audio and audio_dim)
    if have_audio:
        feats = song_features(args.audio, args.bpm, args.offset, n_cells, feats_per_cell)
        print(f"audio: {args.audio} -> {n_cells} cells x {feats_per_cell} features")
    else:
        feats = np.zeros((n_cells + WINDOW, feats_per_cell), dtype=np.float32)

    def window(pos: int) -> np.ndarray:
        c = min(pos // GRID, len(feats) - WINDOW - 1)
        return feats[c:c + WINDOW].reshape(-1)

    radar = {ax: round(max(0.0, min(1.0, getattr(args, ax.replace("-", "_")))) * 200)
             for ax in tok.RADAR_AXES}
    cond = [tok.BOS] + tok.cond_tokens(args.level, radar, args.bpm)
    uncond = [tok.BOS, cond[1]] + [tok.UNCOND] * 6 + [cond[8]]
    gr, ga = args.guidance, args.audio_guidance
    # rows: 0 = radar+audio, 1 = no radar, 2 = no audio (only when audio is on)
    rows_cond = [cond, uncond] + ([cond] if have_audio else [])
    B = len(rows_cond)

    body: list[int] = []
    pos = 0
    print(f"generating (level {args.level}, radar {radar}, bpm {args.bpm:g}, "
          f"guidance {gr}/{ga if have_audio else '-'}) ...")
    with torch.no_grad():
        while len(body) < args.max_tokens and pos < args.measures * tok.MEASURE:
            seqs = [rc + body for rc in rows_cond]
            idx = torch.tensor([s[-model.cfg.ctx:] for s in seqs], device=device)
            au = None
            if audio_dim:
                w = np.stack([window(0)] * tok.PREFIX_LEN
                             + [window(p) for p in _positions(body)])[-model.cfg.ctx:]
                a = np.stack([w] * B)
                if have_audio:
                    a[2] = 0.0  # the audio-free row
                au = torch.from_numpy(a).to(device)
            logits, _ = model(idx, audio=au)
            lg = logits[:, -1, :]
            if have_audio:
                mixed = lg[0] + (gr - 1) * (lg[0] - lg[1]) + (ga - 1) * (lg[0] - lg[2])
            else:
                mixed = lg[1] + gr * (lg[0] - lg[1])
            probs = torch.softmax(mixed / max(1e-6, args.temperature), dim=-1)
            sp, si = torch.sort(probs, descending=True)
            keep = torch.cumsum(sp, 0) - sp < args.top_p
            keep[0] = True
            probs = torch.zeros_like(probs).scatter_(0, si[keep], sp[keep])
            probs /= probs.sum()
            t = int(torch.multinomial(probs, 1).item())
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
    print(f"decoded: {n_notes} notes, {n_pts} laser points, {pos // tok.MEASURE} measures")
    music = os.path.basename(args.audio) if args.audio else ""
    text = tok.chart_to_ksh(
        chart, title=f"generated lv{args.level}", level=args.level, bpm=args.bpm,
        music_file=music, offset_ms=args.offset)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"wrote {args.out}")


def _positions(body: list[int]) -> list[int]:
    out, pos = [], 0
    for t in body:
        name = tok.VOCAB[t]
        if name == "bar":
            pos = (pos // tok.MEASURE + 1) * tok.MEASURE
        elif name.startswith("d_"):
            pos += int(name[2:])
        out.append(pos)
    return out


if __name__ == "__main__":
    main()
