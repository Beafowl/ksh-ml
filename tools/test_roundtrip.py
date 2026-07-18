"""Validate tokenizer + serializer without torch:
  1. record -> tokens -> chart: note/hold/laser data must survive exactly
  2. chart -> .ksh text -> ksh.parse: counts must survive the file format too

Usage: python tools/test_roundtrip.py [--data out/charts.jsonl.gz] [--n 150]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdvx_dataset import ksh  # noqa: E402
from sdvx_model import tokenizer as tok  # noqa: E402


def sig(chart, from_dataset):
    """comparable signature of a chart's playable content"""
    bt = [sorted((n[0], n[1]) for n in lane) for lane in chart["bt"]]
    fx = [sorted((n[0], n[1]) for n in side) for side in chart["fx"]]
    lasers = [sorted((g["wide"], tuple(map(tuple, g["pts"]))) for g in side)
              for side in chart["lasers"]]
    return bt, fx, lasers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "out", "charts.jsonl.gz"))
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    records = []
    with gzip.open(args.data, "rt", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    random.seed(11)
    sample = random.sample(records, min(args.n, len(records)))

    fail = 0
    tok_lens = []
    for r in sample:
        seq = tok.encode(r)
        tok_lens.append(len(seq))
        assert seq[0] == tok.BOS and seq[-1] == tok.EOS
        decoded = tok.decode_body(seq[tok.PREFIX_LEN:-1])
        if sig(r["chart"], True) != sig(decoded, False):
            fail += 1
            print(f"TOKEN ROUNDTRIP MISMATCH: {r['id']}")
            continue
        # full circle through the .ksh serializer and the validated parser
        text = tok.chart_to_ksh(decoded, level=int(r["level"]) or 1,
                                bpm=r["chart"]["bpms"][0][1])
        re = ksh.parse(text)
        re_chart = {
            "bpms": re["bpms"], "sigs": re["sigs"], "bt": re["bt"], "fx": re["fx"],
            "lasers": [[{"wide": g["wide"], "pts": [list(p) for p in g["pts"]]}
                        for g in side] for side in re["lasers"]],
        }
        if sig(decoded, False) != sig(re_chart, False):
            fail += 1
            print(f"KSH ROUNDTRIP MISMATCH: {r['id']}")

    tok_lens.sort()
    print(f"checked {len(sample)} charts: {'ALL OK' if not fail else str(fail) + ' FAILURES'}")
    print(f"token lengths: min {tok_lens[0]}, median {tok_lens[len(tok_lens)//2]}, "
          f"p90 {tok_lens[int(len(tok_lens)*0.9)]}, max {tok_lens[-1]}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
