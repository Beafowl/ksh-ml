"""Precompute dense per-frame log-mel for every charted song (v6 audio).

Reads the v3 charts.jsonl.gz (which carries the resolved `audio` path and
per-chart `end` tick), computes one dense mel per UNIQUE audio file (trimmed
to the longest chart on it), and writes:
  <out>/dense_mels.u8    concatenated uint8 (total_frames, 80), memmap-friendly
  <out>/dense_index.json {chart_id: [frame_offset, n_frames]}

  python -m sdvx_dataset.preprocess_dense \
      --charts out_v3/charts.jsonl.gz --root "C:/Users/Berkb/Downloads/SDVX" \
      --out out_v6 [--workers 8] [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .mel_dense import dense_mel_u8, N_MELS, FPS


def _task(args):
    path, end_ms = args
    try:
        return path, dense_mel_u8(path, end_ms), None
    except Exception as e:  # noqa: BLE001
        return path, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", required=True)
    ap.add_argument("--root", required=True, help="charts root the audio paths are relative to")
    ap.add_argument("--out", default="out_v6")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # unique audio -> (max end_ms, [chart ids])
    audio_ids: dict[str, list[str]] = {}
    audio_end: dict[str, float] = {}
    n = 0
    with gzip.open(args.charts, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("audio"):
                continue
            path = os.path.join(args.root, r["audio"])
            end_tick = r.get("end") or 0
            bpm = (r.get("bpm") or [None, 120])[1] or 120
            end_ms = r.get("offset_ms", 0) + end_tick / 48.0 * (60000.0 / bpm)
            audio_ids.setdefault(path, []).append(r["id"])
            audio_end[path] = max(audio_end.get(path, 0.0), end_ms)
            n += 1
            if args.limit and len(audio_ids) >= args.limit:
                break
    print(f"charts: {n}, unique audio: {len(audio_ids)}, workers: {args.workers}")

    tasks = [(p, audio_end[p]) for p in audio_ids]
    index: dict[str, list[int]] = {}
    offset = 0
    fails = []
    out_path = os.path.join(args.out, "dense_mels.u8")
    with open(out_path, "wb") as fout, \
         ProcessPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        for path, mel, err in ex.map(_task, tasks, chunksize=2):
            done += 1
            if err is not None or mel is None:
                fails.append((path, err))
                continue
            assert mel.shape[1] == N_MELS
            fout.write(mel.tobytes())
            nf = mel.shape[0]
            for cid in audio_ids[path]:
                index[cid] = [offset, nf]
            offset += nf
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)}  frames={offset} ({offset/FPS/3600:.1f}h audio)")

    json.dump({"n_mels": N_MELS, "fps": FPS, "total_frames": offset, "index": index},
              open(os.path.join(args.out, "dense_index.json"), "w"))
    size_gb = offset * N_MELS / 1e9
    print(f"done. {len(index)} charts, {offset} frames, {size_gb:.2f} GB, "
          f"{len(fails)} failures")
    for p, e in fails[:10]:
        print("  FAIL", os.path.basename(p), e)


if __name__ == "__main__":
    main()
