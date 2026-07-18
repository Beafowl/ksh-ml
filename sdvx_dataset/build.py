"""Build the training corpus: match converted KSH charts to official radar
labels from music_db.xml, parse every chart, extract per-1/16 onset features
from the audio, and write one JSONL line per chart plus a sanity report.

Usage:
  python -m sdvx_dataset.build --charts "C:/Users/Berkb/Downloads/SDVX" \
      --db "D:/Spiele/KFC-2026020300/contents/data/others/music_db.xml" \
      --out out [--no-audio] [--workers 8] [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import ksh, music_db
from .matching import TitleIndex, ksh_max_bpm
from .onsets import FEATS, grid_features, onset_features

GRID_STEP = 12  # ticks per onset cell (1/16 note)

# ksh header difficulty= -> candidate music_db slots
BASE_SLOTS = {
    "light": ["novice"],
    "challenge": ["advanced"],
    "extended": ["exhaust"],
    "infinite": ["infinite", "maximum"],
}
# filename hints that disambiguate the 4th slot
HINT_MAXIMUM = ("mxm",)
HINT_INFINITE = ("inf", "grv", "hvn", "vvd", "xcd", "4g")


def read_ksh_text(path: str) -> str:
    raw = open(path, "rb").read()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp932", errors="replace")


def pick_slot(song, header_diff: str, ksh_level: int | None, stem: str):
    """-> Difficulty | None from the db song for this ksh file."""
    if song is None:
        return None
    cands = [s for s in BASE_SLOTS.get(header_diff, music_db.SLOTS) if s in song.diffs]
    if not cands:
        cands = [s for s in music_db.SLOTS if s in song.diffs]
        if ksh_level is not None:
            lv = [s for s in cands if math.floor(song.diffs[s].level) == ksh_level]
            cands = lv or cands
        return song.diffs[cands[0]] if len(cands) == 1 else None
    if len(cands) > 1:
        st = stem.lower()
        if any(h in st for h in HINT_MAXIMUM) and "maximum" in cands:
            cands = ["maximum"]
        elif any(h in st for h in HINT_INFINITE) and "infinite" in cands:
            cands = ["infinite"]
        elif ksh_level is not None:
            lv = [s for s in cands if math.floor(song.diffs[s].level) == ksh_level]
            if len(lv) == 1:
                cands = lv
    return song.diffs[cands[0]]


def find_audio(song_dir: str, files: list[str], meta: dict, stem: str):
    m = (meta.get("m", "") or "").split(";")[0].strip()
    for cand in (m, stem + ".ogg"):
        if cand and cand in files:
            return cand
    oggs = [f for f in files if f.lower().endswith(".ogg")]
    return oggs[0] if len(oggs) == 1 else None


def collect(charts_root: str, index: TitleIndex, aliases: dict, limit: int | None):
    records, stats = [], Counter()
    unmatched = {}
    parse_errors = []
    for version in sorted(os.listdir(charts_root)):
        vdir = os.path.join(charts_root, version)
        if not os.path.isdir(vdir):
            continue
        for folder in sorted(os.listdir(vdir)):
            sdir = os.path.join(vdir, folder)
            if not os.path.isdir(sdir):
                continue
            if limit and stats["charts"] >= limit:
                return records, stats, unmatched, parse_errors
            files = os.listdir(sdir)
            kshs = sorted(f for f in files if f.lower().endswith(".ksh"))
            if not kshs:
                continue
            stats["songs"] += 1
            song_key = f"{version}/{folder}"

            parsed = []
            for f in kshs:
                try:
                    chart = ksh.parse(read_ksh_text(os.path.join(sdir, f)))
                    parsed.append((f, chart))
                except Exception as e:  # noqa: BLE001 - report and move on
                    stats["parse_errors"] += 1
                    parse_errors.append(f"{song_key}/{f}: {e!r}")
            if not parsed:
                continue

            title = parsed[0][1]["meta"].get("title", "")
            bpm = ksh_max_bpm(parsed[0][1]["meta"].get("t", ""))
            song, how = index.match(title, bpm, aliases.get(song_key))
            stats[f"match_{how}"] += 1
            if song is None:
                unmatched[song_key] = title

            for f, chart in parsed:
                stem = f[:-4]
                meta = chart["meta"]
                try:
                    ksh_level = int(meta.get("level", ""))
                except ValueError:
                    ksh_level = None
                end = ksh.end_tick(chart)
                n_notes = sum(len(l) for l in chart["bt"]) + sum(len(s) for s in chart["fx"])
                n_laser = sum(len(g["pts"]) for s in chart["lasers"] for g in s)
                if end < ksh.WHOLE_TICKS or n_notes + n_laser < 16:
                    stats["skipped_empty"] += 1
                    continue

                diff = pick_slot(song, meta.get("difficulty", ""), ksh_level, stem)
                radar = diff.radar if diff else None
                stats["charts"] += 1
                if radar:
                    stats["labeled"] += 1
                if diff and ksh_level is not None and math.floor(diff.level) != ksh_level:
                    stats["level_mismatch"] += 1

                audio = find_audio(sdir, files, meta, stem)
                if audio:
                    stats["with_audio"] += 1
                records.append({
                    "id": f"{song_key}/{stem}",
                    "music_id": song.music_id if song else None,
                    "title": song.title if song else title,
                    "version_folder": version,
                    "slot": diff.slot if diff else meta.get("difficulty", ""),
                    "level": diff.level if diff else float(ksh_level or 0),
                    "labeled": bool(radar),
                    "radar": radar,
                    "effected_by": diff.effected_by if diff else "",
                    "bpm": [song.bpm_min, song.bpm_max] if song else [bpm or 0, bpm or 0],
                    "offset_ms": float(meta.get("o", "0") or 0),
                    "end": end,
                    "chart": {
                        "bpms": [[y, v] for y, v in chart["bpms"]],
                        "sigs": [list(s) for s in chart["sigs"]],
                        "bt": chart["bt"],
                        "fx": chart["fx"],
                        "lasers": [
                            [{"wide": g["wide"], "pts": [list(p) for p in g["pts"]]} for g in side]
                            for side in chart["lasers"]
                        ],
                    },
                    "audio": f"{song_key}/{audio}" if audio else None,
                    "onset": None,
                    "onset_step": GRID_STEP,
                    "onset_feats": FEATS,
                    "_grid_ms": None if not audio else _grid_ms(chart, end),
                })
    return records, stats, unmatched, parse_errors


def _grid_ms(chart, end):
    t = ksh.Timing(chart)
    return [round(t.tick_to_ms(y), 2) for y in range(0, end + GRID_STEP, GRID_STEP)]


def _audio_task(args):
    path, jobs = args  # jobs: [(chart_id, grid_ms list)]
    out = []
    try:
        feats, fps = onset_features(path)
        for cid, grid in jobs:
            vals = grid_features(feats, fps, np.asarray(grid, dtype=np.float64))
            out.append((cid, np.round(vals, 3).tolist()))
    except Exception as e:  # noqa: BLE001
        out = [(cid, None) for cid, _ in jobs]
        return out, f"{path}: {e!r}"
    return out, None


def run_audio(records, charts_root, workers):
    from tqdm import tqdm
    by_audio = {}
    for r in records:
        if r["audio"] and r["_grid_ms"]:
            path = os.path.join(charts_root, r["audio"])
            by_audio.setdefault(path, []).append((r["id"], r["_grid_ms"]))
    by_id = {r["id"]: r for r in records}
    tasks = list(by_audio.items())
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out, err in tqdm(ex.map(_audio_task, tasks, chunksize=4),
                             total=len(tasks), desc="audio", unit="file"):
            if err:
                failures.append(err)
            for cid, vals in out:
                by_id[cid]["onset"] = vals
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=None, help="max charts (smoke runs)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    aliases = {}
    alias_path = os.path.join(os.path.dirname(__file__), "..", "aliases.json")
    if os.path.exists(alias_path):
        aliases = {k: v for k, v in json.load(open(alias_path, encoding="utf-8")).items()
                   if isinstance(v, int)}

    print("loading music_db ...")
    songs = music_db.load(args.db)
    index = TitleIndex(songs)
    print(f"  {len(songs)} songs, "
          f"{sum(1 for s in songs for d in s.diffs.values() if d.radar)} charts with radar")

    print("parsing charts ...")
    records, stats, unmatched, parse_errors = collect(args.charts, index, aliases, args.limit)

    audio_failures = []
    if not args.no_audio:
        audio_failures = run_audio(records, args.charts, args.workers)

    out_path = os.path.join(args.out, "charts.jsonl.gz")
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for r in sorted(records, key=lambda r: r["id"]):
            r.pop("_grid_ms", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    json.dump({k: None for k in sorted(unmatched)} | {"_titles": unmatched},
              open(os.path.join(args.out, "unmatched.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---- report ----
    labeled = [r for r in records if r["labeled"]]
    lines = ["== dataset report ==",
             f"songs scanned:        {stats['songs']}",
             f"charts kept:          {stats['charts']}  (skipped empty: {stats['skipped_empty']}, parse errors: {stats['parse_errors']})",
             f"matched songs:        exact {stats['match_exact']} / kana {stats['match_kana']} / suffix {stats['match_suffix']} / alias {stats['match_alias']} / none {stats['match_none']}",
             f"charts with radar:    {stats['labeled']} ({100 * stats['labeled'] // max(1, stats['charts'])}%)",
             f"charts with audio:    {stats['with_audio']}, onset extracted: {sum(1 for r in records if r['onset'])}",
             f"level label mismatch: {stats['level_mismatch']} (db level used)",
             f"audio failures:       {len(audio_failures)}"]
    if labeled:
        lines.append("radar axis stats (0-200):")
        for ax in music_db.RADAR_AXES:
            vals = [r["radar"][ax] for r in labeled]
            lines.append(f"  {ax:9s} min {min(vals):3d}  mean {sum(vals) / len(vals):6.1f}  max {max(vals):3d}")
        hist = Counter(math.floor(r["level"]) for r in labeled)
        lines.append("level histogram: " + " ".join(f"{l}:{hist[l]}" for l in sorted(hist)))
    ev = sum(sum(len(l) for l in r["chart"]["bt"]) + sum(len(s) for s in r["chart"]["fx"])
             + sum(len(g["pts"]) for s in r["chart"]["lasers"] for g in s) for r in records)
    lines.append(f"total note/laser events: {ev}")
    if parse_errors:
        lines.append("parse errors (first 10):")
        lines += [f"  {e}" for e in parse_errors[:10]]
    if audio_failures:
        lines.append("audio failures (first 10):")
        lines += [f"  {e}" for e in audio_failures[:10]]
    report = "\n".join(lines)
    open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8").write(report + "\n")
    print(report)
    print(f"\nwrote {out_path} ({os.path.getsize(out_path) // 1024} KB)")


if __name__ == "__main__":
    main()
