"""v6 tokenizer: KShootMania charts as 6-key mania (no lasers).

Columns map the two button banks into one 6-lane space:
  col 0-3 = BT A/B/C/D,  col 4-5 = FX L/R
Lasers are dropped for v6 (a later pass adds them back).

Sequence layout (mirrors the v5 prefix so conditioning is unchanged):
  <bos> lv_X notes_B peak_B tsumami_B tricky_B hand-trip_B one-hand_B bpm_B eff_S
  <body events> <eos>

Body events are delta-time coded on the 48-ticks-per-beat grid:
  bar            advance to the next measure (192 ticks)
  d_N            advance N ticks (N in DELTAS; big gaps sum several)
  note_C         chip in column C (0-5)
  hold_on_C / hold_off_C   hold start / release in column C

The audio side (dense per-frame mel) is handled by the model, not here;
encode_with_ticks still returns per-token tick positions for alignment.
"""
from __future__ import annotations

import math

RADAR_AXES = ["notes", "peak", "tsumami", "tricky", "hand-trip", "one-hand"]
RADAR_BUCKETS = 11
BPM_BUCKETS = 64
DELTAS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 96]
MEASURE = 192
EFF_VOCAB = 100
COLS = 6  # BT A-D + FX L/R


def _build_vocab() -> list[str]:
    v = ["<pad>", "<bos>", "<eos>", "<uncond>"]
    v += [f"lv_{i}" for i in range(1, 21)]
    for ax in RADAR_AXES:
        v += [f"{ax}_{b}" for b in range(RADAR_BUCKETS)]
    v += [f"bpm_{i}" for i in range(BPM_BUCKETS)]
    v += [f"eff_{i}" for i in range(EFF_VOCAB)]
    v += ["bar"] + [f"d_{n}" for n in DELTAS]
    for c in range(COLS):
        v += [f"note_{c}", f"hold_on_{c}", f"hold_off_{c}"]
    return v


VOCAB = _build_vocab()
ID = {name: i for i, name in enumerate(VOCAB)}
PAD, BOS, EOS, UNCOND = ID["<pad>"], ID["<bos>"], ID["<eos>"], ID["<uncond>"]
COND_LEN = 9  # lv + 6 radar + bpm + effector
PREFIX_LEN = 1 + COND_LEN
RADAR_SLOTS = range(2, 8)
EFF_SLOT = 9


def radar_bucket(v: int) -> int:
    return max(0, min(RADAR_BUCKETS - 1, round(v / 200 * (RADAR_BUCKETS - 1))))


def bpm_bucket(bpm: float) -> int:
    bpm = max(50.0, min(400.0, bpm or 120.0))
    x = (math.log2(bpm) - math.log2(50)) / (math.log2(400) - math.log2(50))
    return max(0, min(BPM_BUCKETS - 1, round(x * (BPM_BUCKETS - 1))))


def cond_tokens(level, radar, bpm, eff_id=None) -> list[int]:
    lv = max(1, min(20, int(level) or 1))
    out = [ID[f"lv_{lv}"]]
    for ax in RADAR_AXES:
        out.append(ID[f"{ax}_{radar_bucket(radar[ax])}"] if radar else UNCOND)
    out.append(ID[f"bpm_{bpm_bucket(bpm)}"])
    out.append(ID[f"eff_{eff_id}"] if eff_id is not None and 0 <= eff_id < EFF_VOCAB else UNCOND)
    return out


def _column_events(chart: dict):
    """chart bt/fx -> list of (tick, priority, tokens); lasers ignored."""
    ev = []
    for l, lane in enumerate(chart["bt"]):
        for y, ln in lane:
            if ln > 0:
                ev.append((y, 2, [ID[f"hold_on_{l}"]]))
                ev.append((y + ln, 1, [ID[f"hold_off_{l}"]]))
            else:
                ev.append((y, 3, [ID[f"note_{l}"]]))
    for s, side in enumerate(chart["fx"]):
        for note in side:
            y, ln = note[0], note[1]
            c = 4 + s
            if ln > 0:
                ev.append((y, 2, [ID[f"hold_on_{c}"]]))
                ev.append((y + ln, 1, [ID[f"hold_off_{c}"]]))
            else:
                ev.append((y, 3, [ID[f"note_{c}"]]))
    return ev


def encode_body(chart: dict):
    ev = _column_events(chart)
    ev.sort(key=lambda e: (e[0], e[1]))
    out, ticks = [], []
    pos = 0
    for tick, _, tokens in ev:
        while pos < tick:
            next_bar = (pos // MEASURE + 1) * MEASURE
            if tick >= next_bar:
                pos = next_bar
                out.append(ID["bar"])
            else:
                gap = tick - pos
                d = max(n for n in DELTAS if n <= gap)
                pos += d
                out.append(ID[f"d_{d}"])
            ticks.append(pos)
        for t in tokens:
            out.append(t)
            ticks.append(pos)
    return out, ticks


def encode_with_ticks(record: dict, eff_map=None):
    bpm = record["bpm"][1] or record["chart"]["bpms"][0][1]
    radar = record["radar"] if record.get("labeled") else None
    eff_id = eff_map.get(record.get("effected_by") or "") if eff_map else None
    body, bticks = encode_body(record["chart"])
    tokens = [BOS] + cond_tokens(record["level"], radar, bpm, eff_id) + body + [EOS]
    ticks = [0] * PREFIX_LEN + bticks + [bticks[-1] if bticks else 0]
    return tokens, ticks


def decode_body(tokens: list[int]) -> dict:
    """Body tokens -> chart dict (bt/fx only; lasers empty)."""
    bt = [[] for _ in range(4)]
    fx = [[] for _ in range(2)]
    hold_open = [None] * COLS
    pos = 0
    for t in tokens:
        name = VOCAB[t] if 0 <= t < len(VOCAB) else "<pad>"
        if name == "bar":
            pos = (pos // MEASURE + 1) * MEASURE
        elif name.startswith("d_"):
            pos += int(name[2:])
        elif name.startswith("note_"):
            c = int(name[-1])
            (bt[c] if c < 4 else fx[c - 4]).append([pos, 0] if c < 4 else [pos, 0, ""])
        elif name.startswith("hold_on_"):
            c = int(name[-1])
            if hold_open[c] is None:
                obj = [pos, 0] if c < 4 else [pos, 0, ""]
                hold_open[c] = obj
                (bt[c] if c < 4 else fx[c - 4]).append(obj)
        elif name.startswith("hold_off_"):
            c = int(name[-1])
            if hold_open[c] is not None:
                hold_open[c][1] = max(0, pos - hold_open[c][0])
                hold_open[c] = None
    for c in range(COLS):
        if hold_open[c] is not None:
            hold_open[c][1] = max(0, pos - hold_open[c][0])
    return {"bpms": [[0, 120.0]], "sigs": [[0, 4, 4]], "bt": bt, "fx": fx,
            "lasers": [[], []]}
