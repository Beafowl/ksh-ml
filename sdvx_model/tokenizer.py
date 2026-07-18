"""Chart <-> token sequence, plus a chart -> .ksh serializer for sampling.

Sequence layout:
  <bos> lv_X notes_B peak_B tsumami_B tricky_B hand-trip_B one-hand_B bpm_B
  <body events> <eos>

Body events are delta-time coded on the 48-ticks-per-beat grid:
  bar          advance to the next multiple of 192 ticks (measure boundary)
  d_N          advance N ticks (N in DELTAS; larger gaps sum several tokens)
  bt_chip_L / bt_on_L / bt_off_L      BT chip / hold start / hold end, lane 0-3
  fx_chip_S / fx_on_S / fx_off_S      FX, side 0-1
  la_on_S [la_wide_S] la_v_S_V ... la_off_S   laser segment, V = 0..50
  bpmch bpm_B                          BPM change at the current tick

Unlabeled charts use <uncond> in the radar slots — the same token that
classifier-free-guidance dropout uses during training.
"""
from __future__ import annotations

import math

RADAR_AXES = ["notes", "peak", "tsumami", "tricky", "hand-trip", "one-hand"]
RADAR_BUCKETS = 11        # 0-200 -> 0-10
BPM_BUCKETS = 64          # log scale 50..400
DELTAS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 96]
MEASURE = 192


def _build_vocab() -> list[str]:
    v = ["<pad>", "<bos>", "<eos>", "<uncond>"]
    v += [f"lv_{i}" for i in range(1, 21)]
    for ax in RADAR_AXES:
        v += [f"{ax}_{b}" for b in range(RADAR_BUCKETS)]
    v += [f"bpm_{i}" for i in range(BPM_BUCKETS)]
    v += ["bar"] + [f"d_{n}" for n in DELTAS]
    for l in range(4):
        v += [f"bt_chip_{l}", f"bt_on_{l}", f"bt_off_{l}"]
    for s in range(2):
        v += [f"fx_chip_{s}", f"fx_on_{s}", f"fx_off_{s}"]
    for s in range(2):
        v += [f"la_on_{s}", f"la_wide_{s}", f"la_off_{s}"]
        v += [f"la_v_{s}_{i}" for i in range(51)]
    v += ["bpmch"]
    return v


VOCAB = _build_vocab()
ID = {name: i for i, name in enumerate(VOCAB)}
PAD, BOS, EOS, UNCOND = ID["<pad>"], ID["<bos>"], ID["<eos>"], ID["<uncond>"]
COND_LEN = 8  # lv + 6 radar + bpm
PREFIX_LEN = 1 + COND_LEN
RADAR_SLOTS = range(2, 8)  # positions of the radar tokens within a sequence


def radar_bucket(v: int) -> int:
    return max(0, min(RADAR_BUCKETS - 1, round(v / 200 * (RADAR_BUCKETS - 1))))


def bpm_bucket(bpm: float) -> int:
    bpm = max(50.0, min(400.0, bpm or 120.0))
    x = (math.log2(bpm) - math.log2(50)) / (math.log2(400) - math.log2(50))
    return max(0, min(BPM_BUCKETS - 1, round(x * (BPM_BUCKETS - 1))))


def cond_tokens(level: float, radar: dict | None, bpm: float) -> list[int]:
    lv = max(1, min(20, int(level) or 1))
    out = [ID[f"lv_{lv}"]]
    for ax in RADAR_AXES:
        out.append(ID[f"{ax}_{radar_bucket(radar[ax])}"] if radar else UNCOND)
    out.append(ID[f"bpm_{bpm_bucket(bpm)}"])
    return out


def encode_body(chart: dict) -> list[int]:
    """chart: the dataset record's 'chart' dict (bpms/bt/fx/lasers)."""
    ev: list[tuple[int, int, list[int]]] = []  # (tick, priority, tokens)
    for y, v in chart["bpms"][1:]:
        ev.append((y, 0, [ID["bpmch"], ID[f"bpm_{bpm_bucket(v)}"]]))
    for l, lane in enumerate(chart["bt"]):
        for y, ln in lane:
            if ln > 0:
                ev.append((y, 2, [ID[f"bt_on_{l}"]]))
                ev.append((y + ln, 1, [ID[f"bt_off_{l}"]]))
            else:
                ev.append((y, 3, [ID[f"bt_chip_{l}"]]))
    for s, side in enumerate(chart["fx"]):
        for note in side:
            y, ln = note[0], note[1]
            if ln > 0:
                ev.append((y, 2, [ID[f"fx_on_{s}"]]))
                ev.append((y + ln, 1, [ID[f"fx_off_{s}"]]))
            else:
                ev.append((y, 3, [ID[f"fx_chip_{s}"]]))
    for s, side in enumerate(chart["lasers"]):
        for seg in side:
            pts = seg["pts"]
            start = [ID[f"la_on_{s}"]] + ([ID[f"la_wide_{s}"]] if seg["wide"] == 2 else [])
            ev.append((pts[0][0], 4, start))
            for y, v in pts:
                ev.append((y, 5, [ID[f"la_v_{s}_{v}"]]))
            ev.append((pts[-1][0], 6, [ID[f"la_off_{s}"]]))
    ev.sort(key=lambda e: (e[0], e[1]))
    out: list[int] = []
    ticks: list[int] = []  # tick position AFTER each token (audio alignment)
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
    return (out, ticks)


def encode(record: dict) -> list[int]:
    tokens, _ = encode_with_ticks(record)
    return tokens


def encode_with_ticks(record: dict) -> tuple[list[int], list[int]]:
    """-> (tokens, tick-after-token); prefix/EOS positions are 0/end."""
    bpm = record["bpm"][1] or record["chart"]["bpms"][0][1]
    radar = record["radar"] if record.get("labeled") else None
    body, bticks = encode_body(record["chart"])
    tokens = [BOS] + cond_tokens(record["level"], radar, bpm) + body + [EOS]
    ticks = [0] * PREFIX_LEN + bticks + [bticks[-1] if bticks else 0]
    return tokens, ticks


def decode_body(tokens: list[int]) -> dict:
    """Token ids (body only, specials tolerated) -> chart dict. Malformed
    structures are repaired: unclosed holds/lasers are closed at the end."""
    bt = [[] for _ in range(4)]
    fx = [[] for _ in range(2)]
    lasers: list[list] = [[], []]
    bpms: list[list] = []
    bt_open: list = [None] * 4
    fx_open: list = [None] * 2
    la_open: list = [None, None]
    pos = 0
    pending_bpm = False
    for t in tokens:
        name = VOCAB[t] if 0 <= t < len(VOCAB) else "<pad>"
        if pending_bpm:
            pending_bpm = False
            if name.startswith("bpm_"):
                i = int(name[4:])
                x = i / (BPM_BUCKETS - 1)
                bpms.append([pos, round(2 ** (math.log2(50) + x * (math.log2(400) - math.log2(50))), 2)])
                continue
        if name == "bar":
            pos = (pos // MEASURE + 1) * MEASURE
        elif name.startswith("d_"):
            pos += int(name[2:])
        elif name == "bpmch":
            pending_bpm = True
        elif name.startswith("bt_chip_"):
            bt[int(name[-1])].append([pos, 0])
        elif name.startswith("bt_on_"):
            l = int(name[-1])
            if bt_open[l] is None:
                bt_open[l] = [pos, 0]
                bt[l].append(bt_open[l])
        elif name.startswith("bt_off_"):
            l = int(name[-1])
            if bt_open[l] is not None:
                bt_open[l][1] = max(0, pos - bt_open[l][0])
                bt_open[l] = None
        elif name.startswith("fx_chip_"):
            fx[int(name[-1])].append([pos, 0, ""])
        elif name.startswith("fx_on_"):
            s = int(name[-1])
            if fx_open[s] is None:
                fx_open[s] = [pos, 0, ""]
                fx[s].append(fx_open[s])
        elif name.startswith("fx_off_"):
            s = int(name[-1])
            if fx_open[s] is not None:
                fx_open[s][1] = max(0, pos - fx_open[s][0])
                fx_open[s] = None
        elif name.startswith("la_on_"):
            s = int(name[-1])
            la_open[s] = {"wide": 1, "pts": []}
        elif name.startswith("la_wide_"):
            s = int(name[-1])
            if la_open[s] is not None:
                la_open[s]["wide"] = 2
        elif name.startswith("la_v_"):
            _, _, s, v = name.split("_")
            s = int(s)
            if la_open[s] is None:
                la_open[s] = {"wide": 1, "pts": []}
            pts = la_open[s]["pts"]
            if pts and pts[-1][0] == pos:
                pts[-1][1] = int(v)  # duplicate value at one tick: keep last
            else:
                pts.append([pos, int(v)])
        elif name.startswith("la_off_"):
            s = int(name[-1])
            if la_open[s] is not None:
                if len(la_open[s]["pts"]) >= 2:
                    lasers[s].append(la_open[s])
                la_open[s] = None
    # repair: close dangling structures
    for l in range(4):
        if bt_open[l] is not None:
            bt_open[l][1] = max(0, pos - bt_open[l][0])
    for s in range(2):
        if fx_open[s] is not None:
            fx_open[s][1] = max(0, pos - fx_open[s][0])
        if la_open[s] is not None and len(la_open[s]["pts"]) >= 2:
            lasers[s].append(la_open[s])
    if not bpms or bpms[0][0] > 0:
        bpms.insert(0, [0, 120.0])
    return {"bpms": bpms, "sigs": [[0, 4, 4]], "bt": bt, "fx": fx, "lasers": lasers}


# ------------------------- chart -> .ksh -------------------------

LASER_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmno"


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def chart_to_ksh(chart: dict, title="generated", difficulty="infinite",
                 level=15, bpm=None, music_file="", offset_ms=0) -> str:
    bpms = chart["bpms"]
    if bpm is not None:
        bpms = [[0, float(bpm)]] + [b for b in bpms[1:]]
    end = 0
    for lane in chart["bt"]:
        for n in lane:
            end = max(end, n[0] + n[1])
    for side in chart["fx"]:
        for n in side:
            end = max(end, n[0] + n[1])
    for side in chart["lasers"]:
        for g in side:
            if g["pts"]:
                end = max(end, g["pts"][-1][0])
    n_measures = max(1, -(-(end + 1) // MEASURE))

    bt_chip = [{n[0] for n in lane if n[1] == 0} for lane in chart["bt"]]
    bt_hold = [[(n[0], n[0] + n[1]) for n in lane if n[1] > 0] for lane in chart["bt"]]
    fx_chip = [{n[0] for n in side if n[1] == 0} for side in chart["fx"]]
    fx_hold = [[(n[0], n[0] + n[1]) for n in side if n[1] > 0] for side in chart["fx"]]
    la_pts = [{}, {}]
    la_span = [[], []]
    la_wide_start = [{}, {}]
    for s in range(2):
        for g in chart["lasers"][s]:
            for y, v in g["pts"]:
                la_pts[s][y] = v
            la_span[s].append((g["pts"][0][0], g["pts"][-1][0]))
            if g["wide"] == 2:
                la_wide_start[s][g["pts"][0][0]] = True
    bpm_at = {y: v for y, v in bpms if y > 0}

    # adjacent same-side segments need a '-' row strictly between them or the
    # parser merges them; when no measure boundary lies in the gap, the measure
    # holding the gap must be subdivided finely enough to fit that row
    sep_need: dict[int, int] = {}  # measure index -> min gap that needs a row
    for s in range(2):
        spans_sorted = sorted(la_span[s])
        for (_, a1), (b0, _) in zip(spans_sorted, spans_sorted[1:]):
            e, st = a1, b0
            if st - e <= 1:
                continue  # degenerate; nothing can separate them
            first_bar = (e // MEASURE + 1) * MEASURE
            if first_bar < st:
                continue  # a measure-boundary row already separates them
            m = e // MEASURE
            sep_need[m] = min(sep_need.get(m, MEASURE + 1), st - e)

    def in_span(spans, t, end_incl=False):
        return any(a <= t <= b if end_incl else a <= t < b for a, b in spans)

    head_bpm = bpms[0][1]
    lines = [
        f"title={title}", "artist=sdvx-ml", "effect=sdvx-ml", "jacket=",
        "illustrator=", f"difficulty={difficulty}", f"level={max(1, min(20, int(level)))}",
        f"t={head_bpm:g}", f"m={music_file}", "mvol=75", f"o={int(offset_ms)}", "bg=desert",
        "layer=arrow", "po=0", "plength=15000", "pfiltergain=50", "filtertype=peak",
        "chokkakuautovol=0", "chokkakuvol=50", "ver=171", "--",
    ]
    for mi in range(n_measures):
        m0 = mi * MEASURE
        ticks = set()
        for s in range(4):
            ticks |= {t - m0 for t in bt_chip[s] if m0 <= t < m0 + MEASURE}
            for a, b in bt_hold[s]:
                for t in (a, b):
                    if m0 <= t < m0 + MEASURE:
                        ticks.add(t - m0)
        for s in range(2):
            ticks |= {t - m0 for t in fx_chip[s] if m0 <= t < m0 + MEASURE}
            for a, b in fx_hold[s]:
                for t in (a, b):
                    if m0 <= t < m0 + MEASURE:
                        ticks.add(t - m0)
            ticks |= {t - m0 for t in la_pts[s] if m0 <= t < m0 + MEASURE}
        ticks |= {t - m0 for t in bpm_at if m0 <= t < m0 + MEASURE}
        step = MEASURE
        for t in ticks:
            step = _gcd(step, t)
        step = max(1, step)
        need = sep_need.get(mi)
        if need is not None and step >= need:
            # largest divisor of step that still fits a separator row in the gap
            step = max((d for d in range(1, step) if step % d == 0 and d < need),
                       default=1)
        rows = MEASURE // step

        for r in range(rows):
            t = m0 + r * step
            if t in bpm_at:
                lines.append(f"t={bpm_at[t]:g}")
            for s in range(2):
                if la_wide_start[s].get(t):
                    lines.append(f"laserrange_{'lr'[s]}=2x")
            row = ""
            for l in range(4):
                if in_span(bt_hold[l], t):
                    row += "2"
                elif t in bt_chip[l]:
                    row += "1"
                else:
                    row += "0"
            row += "|"
            for s in range(2):
                if in_span(fx_hold[s], t):
                    row += "1"
                elif t in fx_chip[s]:
                    row += "2"
                else:
                    row += "0"
            row += "|"
            for s in range(2):
                if t in la_pts[s]:
                    row += LASER_CHARS[max(0, min(50, la_pts[s][t]))]
                elif in_span(la_span[s], t, end_incl=True):
                    row += ":"
                else:
                    row += "-"
            lines.append(row)
        lines.append("--")
    return "\r\n".join(lines) + "\r\n"
