"""Cross-check the Python KSH parser against the editor's ksh.js on a random
sample of converted charts: note/laser/bpm counts and end tick must agree.

Usage: python tools/validate_parser.py --charts <root> [--n 40]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdvx_dataset import ksh  # noqa: E402
from sdvx_dataset.build import read_ksh_text  # noqa: E402

NODE_SNIPPET = r"""
const KSH = require(process.argv[1]);
const fs = require("fs");
const files = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = [];
for (const f of files) {
  const raw = fs.readFileSync(f);
  let txt; try { txt = new TextDecoder("utf-8", {fatal: true}).decode(raw); }
  catch (e) { txt = new TextDecoder("shift-jis").decode(raw); }
  const c = KSH.parse(txt);
  out.push({
    bt: c.bt.map(l => l.length), btHold: c.bt.map(l => l.filter(n => n.l > 0).length),
    fx: c.fx.map(s => s.length), fxHold: c.fx.map(s => s.filter(n => n.l > 0).length),
    laserSegs: c.lasers.map(s => s.length),
    laserPts: c.lasers.map(s => s.reduce((a, g) => a + g.points.length, 0)),
    bpms: c.bpms.length, sigs: c.sigs.length, end: KSH.endTick(c),
  });
}
console.log(JSON.stringify(out));
"""


def summarize_py(path):
    c = ksh.parse(read_ksh_text(path))
    end = ksh.end_tick(c)
    return {
        "bt": [len(l) for l in c["bt"]],
        "btHold": [sum(1 for n in l if n[1] > 0) for l in c["bt"]],
        "fx": [len(s) for s in c["fx"]],
        "fxHold": [sum(1 for n in s if n[1] > 0) for s in c["fx"]],
        "laserSegs": [len(s) for s in c["lasers"]],
        "laserPts": [sum(len(g["pts"]) for g in s) for s in c["lasers"]],
        "bpms": len(c["bpms"]), "sigs": len(c["sigs"]), "end": end,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", required=True)
    ap.add_argument("--ksh-js", default=os.path.join(os.path.dirname(__file__), "..", "..", "ksm-editor", "ksh.js"))
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    all_ksh = []
    for root, _, files in os.walk(args.charts):
        for f in files:
            if f.lower().endswith(".ksh"):
                all_ksh.append(os.path.join(root, f))
    random.seed(7)
    sample = random.sample(all_ksh, min(args.n, len(all_ksh)))

    list_path = os.path.join(os.path.dirname(__file__), "_sample.json")
    json.dump(sample, open(list_path, "w"))
    node_out = subprocess.run(
        ["node", "-e", NODE_SNIPPET, os.path.abspath(args.ksh_js), list_path],
        capture_output=True, text=True, check=True).stdout
    js_results = json.loads(node_out)
    os.remove(list_path)

    # endTick differs by design: ksh.js also counts spins/filters/other lines.
    # Compare it as js >= py and never by more than a couple of measures.
    fails = 0
    for path, js in zip(sample, js_results):
        py = summarize_py(path)
        for key in ("bt", "btHold", "fx", "fxHold", "laserSegs", "laserPts", "bpms", "sigs"):
            if py[key] != js[key]:
                fails += 1
                print(f"MISMATCH {key}: {path}\n  py={py[key]} js={js[key]}")
        if not (js["end"] >= py["end"] and js["end"] - py["end"] <= 4 * ksh.WHOLE_TICKS):
            fails += 1
            print(f"MISMATCH end: {path}\n  py={py['end']} js={js['end']}")
    print(f"checked {len(sample)} charts: {'ALL OK' if fails == 0 else str(fails) + ' mismatching fields'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
