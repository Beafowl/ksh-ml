"""Generate a chart from a v6 checkpoint via windowed encoder-decoder.

  python -m sdvx_model.sample_v6 --ckpt runs/v6/best.pt --audio song.ogg \
      --out gen.ksh --level 16 --bpm 160 --offset 0 --measures 40 \
      --guidance 2.0 --audio-guidance 2.0 [--style NAME]

Slides ~16 s audio windows with 50% overlap: the encoder builds per-window
memory, the decoder generates the events in the window (the part already
produced by earlier windows is prefilled as context), cross-attending audio.
Kept intentionally simple/debuggable — the browser runtime mirrors it once
the approach is validated in Python.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from .model_v6 import ChartV6, ConfigV6
from . import tokenizer_v6 as tk
from .tokenizer import chart_to_ksh
from sdvx_dataset.mel_dense import dense_mel_u8

FPS = 100.0
WINDOW_FRAMES = 1600
WINDOW_MS = WINDOW_FRAMES / FPS * 1000.0
STEP_MS = WINDOW_MS / 2.0


def top_p_sample(logits, temp, top_p, banned):
    logits = logits.clone()
    for b in banned:
        logits[b] = -1e9
    probs = F.softmax(logits / max(1e-6, temp), dim=-1)
    sp, si = torch.sort(probs, descending=True)
    keep = torch.cumsum(sp, 0) - sp < top_p
    keep[0] = True
    probs = torch.zeros_like(probs).scatter_(0, si[keep], sp[keep])
    return int(torch.multinomial(probs / probs.sum(), 1))


def deltas_from(start_tick, abs_events):
    """(tick, token) pairs -> delta/bar-coded token stream from start_tick."""
    out, pos = [], start_tick
    for tick, tok in abs_events:
        while pos < tick:
            nb = (pos // tk.MEASURE + 1) * tk.MEASURE
            if tick >= nb:
                pos = nb; out.append(tk.ID["bar"])
            else:
                d = max(n for n in tk.DELTAS if n <= tick - pos)
                pos += d; out.append(tk.ID[f"d_{d}"])
        out.append(tok)
    return out


def replay_pos(start_tick, toks):
    pos = start_tick
    for t in toks:
        name = tk.VOCAB[t]
        if name == "bar":
            pos = (pos // tk.MEASURE + 1) * tk.MEASURE
        elif name.startswith("d_"):
            pos += int(name[2:])
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default="gen_v6.ksh")
    ap.add_argument("--level", type=int, default=16)
    for ax in tk.RADAR_AXES:
        ap.add_argument(f"--{ax}", type=float, default=0.5)
    ap.add_argument("--bpm", type=float, default=160.0)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--measures", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--audio-guidance", type=float, default=2.0)
    ap.add_argument("--style", default=None)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location=device)
    if ck["vocab"] != tk.VOCAB:
        raise SystemExit("checkpoint vocab mismatch")
    model = ChartV6(ConfigV6(**ck["model_cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])

    eff_id = None
    if args.style:
        names = ck.get("eff_names") or []
        hits = [i for i, n in enumerate(names) if args.style.lower() in n.lower()]
        if hits:
            eff_id = hits[0]; print(f"style: {names[eff_id]}")

    radar = {ax: round(max(0.0, min(1.0, getattr(args, ax.replace("-", "_")))) * 200)
             for ax in tk.RADAR_AXES}
    cond = [tk.BOS] + tk.cond_tokens(args.level, radar, args.bpm, eff_id)
    uncond = [tk.BOS, cond[1]] + [tk.UNCOND] * 6 + [cond[8], cond[9]]
    gr, ga = args.guidance, args.audio_guidance

    end_ms = args.offset + args.measures * tk.MEASURE / 48.0 * (60000.0 / args.bpm)
    mel_full = dense_mel_u8(args.audio, end_ms).astype(np.float32) / 255.0
    n_frames = mel_full.shape[0]
    ms_per_tick = 60000.0 / args.bpm / 48.0
    stop_tick = args.measures * tk.MEASURE

    EFF_BAN = {tk.PAD, tk.BOS, tk.UNCOND} | {tk.ID[f"eff_{i}"] for i in range(tk.EFF_VOCAB)}
    events, covered_tick, w_ms = [], 0, 0.0
    print(f"generating {args.measures} measures over {n_frames/FPS:.0f}s audio ...")
    while covered_tick < stop_tick:
        f0 = int(round((args.offset + w_ms) / 1000.0 * FPS))
        win = np.zeros((WINDOW_FRAMES, mel_full.shape[1]), np.float32)
        a, b = max(0, f0), min(f0 + WINDOW_FRAMES, n_frames)
        if b > a:
            win[a - f0:b - f0] = mel_full[a:b]
        mel = torch.from_numpy(win)[None].to(device)

        start_tick = int(round(w_ms / ms_per_tick))
        win_end_tick = int(round((w_ms + WINDOW_MS) / ms_per_tick))
        pre_events = [(t, tok) for (t, tok) in events if start_tick <= t < covered_tick]
        pre_toks = deltas_from(start_tick, pre_events)

        with torch.no_grad():
            B = 3  # cond+audio, uncond-radar, cond-no-audio
            memory = model.encode(mel).repeat(B, 1, 1)
            memory[2] = 0.0
            cross = model.cross_cache(memory)
            idx = torch.tensor([r + pre_toks for r in (cond, uncond, cond)], device=device)
            logits_all, past = model.decode_step(idx, cross, None, 0)
            fed = idx.shape[1]
            pos_ptr = replay_pos(start_tick, pre_toks)
            while fed < model.cfg.max_tgt:
                lg = logits_all[:, -1, :]
                mixed = lg[0] + (gr - 1) * (lg[0] - lg[1]) + (ga - 1) * (lg[0] - lg[2])
                t = top_p_sample(mixed, args.temperature, args.top_p, EFF_BAN)
                name = tk.VOCAB[t]
                if t == tk.EOS:
                    break
                if name == "bar":
                    pos_ptr = (pos_ptr // tk.MEASURE + 1) * tk.MEASURE
                elif name.startswith("d_"):
                    pos_ptr += int(name[2:])
                elif covered_tick <= pos_ptr < stop_tick:
                    events.append((pos_ptr, t))
                if pos_ptr >= win_end_tick or pos_ptr >= stop_tick:
                    break
                logits_all, past = model.decode_step(
                    torch.full((B, 1), t, device=device), cross, past, fed)
                fed += 1
        covered_tick = min(win_end_tick, stop_tick)
        w_ms += STEP_MS
        print(f"  window @ measure {start_tick // tk.MEASURE}, events={len(events)}")

    events.sort()
    body = deltas_from(0, events)
    chart = tk.decode_body(body)
    n = sum(len(l) for l in chart["bt"]) + sum(len(s) for s in chart["fx"])
    print(f"decoded {n} notes")
    text = chart_to_ksh(chart, title=f"gen v6 lv{args.level}", level=args.level,
                        bpm=args.bpm, music_file=os.path.basename(args.audio),
                        offset_ms=args.offset)
    open(args.out, "w", encoding="utf-8", newline="").write(text)
    print(f"wrote {args.out}  ({n} notes, {args.measures} measures)")


if __name__ == "__main__":
    main()
