"""KSH chart parser — a faithful Python port of the editor's ksh.js.

Chart model (ticks: 48/beat, 192 per 4/4 measure):
  meta:   {key: str}
  bpms:   [(y, bpm)]           bpms[0].y == 0
  sigs:   [(y, n, d)]          time signatures at measure starts
  bt:     [lane0..lane3]       lane = [(y, l)]; l=0 chip, l>0 hold
  fx:     [side0, side1]       side = [(y, l, fx_str)]
  lasers: [side0, side1]       side = [{"wide": 1|2, "pts": [(y, v0_50)]}]
"""
from __future__ import annotations

import math
import re

LASER_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmno"  # v = idx/50
SLAM_TICKS = 6
WHOLE_TICKS = 192
TICKS_PER_BEAT = 48

LEGACY_FX = {
    "S": "Retrigger;8", "V": "Retrigger;12", "T": "Retrigger;16", "W": "Retrigger;24",
    "U": "Retrigger;32", "G": "Gate;4", "H": "Gate;8", "K": "Gate;12", "I": "Gate;16",
    "L": "Gate;24", "J": "Gate;32", "F": "Flanger", "P": "PitchShift;12",
    "B": "BitCrusher;5", "Q": "Phaser", "X": "Wobble;12", "A": "TapeStop", "D": "SideChain",
}

ROW_RE = re.compile(r"^[0-2]{4}\|[0-9A-Za-z]{2}\|[0-9A-Za-z\-:]{2}")


def jsround(x: float) -> int:
    """JS Math.round: halves round up (Python's round() banker-rounds)."""
    return math.floor(x + 0.5)


def parse(text: str) -> dict:
    lines = re.split(r"\r?\n", text)
    i = 0

    meta: dict[str, str] = {}
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "--":
            i += 1
            break
        eq = ln.find("=")
        if eq > 0:
            meta[ln[:eq]] = ln[eq + 1:]
        i += 1

    try:
        head_bpm = float(meta.get("t", ""))
    except ValueError:
        head_bpm = float("nan")

    bpms: list[tuple[int, float]] = []
    sigs: list[tuple[int, int, int]] = []
    bt: list[list] = [[], [], [], []]
    fx: list[list] = [[], []]
    lasers: list[list] = [[], []]

    sig_n, sig_d = 4, 4
    pending_sig: tuple[int, int] | None = None
    tick = 0
    bt_act: list = [None, None, None, None]   # [note, end]
    fx_act: list = [None, None]
    ls_act: list = [None, None]
    pend_wide = [False, False]
    cur_fx = ["", ""]

    def int_or(s, dflt):
        try:
            return int(s)
        except ValueError:
            return dflt

    while i < len(lines):
        m_lines = []
        saw_end = False
        while i < len(lines):
            ln = lines[i].strip()
            i += 1
            if ln == "--":
                saw_end = True
                break
            if ln == "":
                continue
            m_lines.append(ln)
        if not m_lines:
            if not saw_end:
                break
            continue

        # beat= before the first note row applies to THIS measure
        for ln in m_lines:
            if ROW_RE.match(ln):
                break
            if ln.startswith("beat="):
                p = ln[5:].split("/")
                pending_sig = (int_or(p[0], 4), int_or(p[1] if len(p) > 1 else "", 4))
        if pending_sig:
            if pending_sig != (sig_n, sig_d) or tick == 0:
                sig_n, sig_d = pending_sig
                sigs.append((tick, sig_n, sig_d))
            pending_sig = None
        if tick == 0 and not sigs:
            sigs.append((0, 4, 4))

        m_ticks = jsround(WHOLE_TICKS * sig_n / sig_d)
        row_count = sum(1 for ln in m_lines if ROW_RE.match(ln))
        step = m_ticks / row_count if row_count else m_ticks

        r = 0
        for ln in m_lines:
            row_tick = tick + jsround(r * step)
            next_tick = tick + jsround((r + 1) * step)

            if ROW_RE.match(ln):
                bt_s, fx_s, ls_s, rest = ln[0:4], ln[5:7], ln[8:10], ln[10:]

                for l in range(4):
                    c = bt_s[l]
                    if c == "2":
                        if not bt_act[l]:
                            note = [row_tick, 0]
                            bt[l].append(note)
                            bt_act[l] = [note, next_tick]
                        bt_act[l][1] = next_tick
                    else:
                        if bt_act[l]:
                            bt_act[l][0][1] = row_tick - bt_act[l][0][0]
                            bt_act[l] = None
                        if c == "1":
                            bt[l].append([row_tick, 0])

                for s in range(2):
                    c = fx_s[s]
                    if c in ("0", "2"):
                        if fx_act[s]:
                            fx_act[s][0][1] = row_tick - fx_act[s][0][0]
                            fx_act[s] = None
                        if c == "2":
                            fx[s].append([row_tick, 0, ""])
                    else:  # '1' or legacy letter => hold unit
                        if not fx_act[s]:
                            eff = cur_fx[s]
                            if c != "1" and c in LEGACY_FX:
                                eff = LEGACY_FX[c]
                            note = [row_tick, 0, eff]
                            fx[s].append(note)
                            fx_act[s] = [note, next_tick]
                        fx_act[s][1] = next_tick

                for s in range(2):
                    c = ls_s[s]
                    if c == "-":
                        ls_act[s] = None
                    elif c == ":":
                        pass  # interpolation, keep segment alive
                    else:
                        idx = LASER_CHARS.find(c)
                        if idx >= 0:
                            if not ls_act[s]:
                                ls_act[s] = {"wide": 2 if pend_wide[s] else 1, "pts": []}
                                pend_wide[s] = False
                                lasers[s].append(ls_act[s])
                            ls_act[s]["pts"].append((row_tick, idx))
                r += 1
            else:
                eq = ln.find("=")
                if eq > 0:
                    k, v = ln[:eq], ln[eq + 1:]
                    if k == "t":
                        try:
                            b = float(v)
                            if math.isfinite(b) and b > 0:
                                bpms.append((row_tick, b))
                        except ValueError:
                            pass
                    elif k == "beat":
                        if r > 0:
                            p = v.split("/")
                            pending_sig = (int_or(p[0], 4), int_or(p[1] if len(p) > 1 else "", 4))
                    elif k == "fx-l":
                        cur_fx[0] = v
                    elif k == "fx-r":
                        cur_fx[1] = v
                    elif k == "laserrange_l":
                        pend_wide[0] = v.strip() == "2x"
                    elif k == "laserrange_r":
                        pend_wide[1] = v.strip() == "2x"
        tick += m_ticks

    # close dangling holds
    for l in range(4):
        if bt_act[l]:
            bt_act[l][0][1] = bt_act[l][1] - bt_act[l][0][0]
    for s in range(2):
        if fx_act[s]:
            fx_act[s][0][1] = fx_act[s][1] - fx_act[s][0][0]
    # drop degenerate laser segments
    for s in range(2):
        lasers[s] = [g for g in lasers[s] if len(g["pts"]) >= 2]

    if not bpms or bpms[0][0] > 0:
        v = head_bpm if math.isfinite(head_bpm) and head_bpm > 0 else 120.0
        bpms.insert(0, (0, v))
    bpms.sort(key=lambda b: b[0])
    bpms = [b for i2, b in enumerate(bpms) if i2 == len(bpms) - 1 or bpms[i2 + 1][0] != b[0]]
    if not sigs:
        sigs = [(0, 4, 4)]

    for lane in bt:
        lane.sort(key=lambda n: n[0])
    for side in fx:
        side.sort(key=lambda n: n[0])
    for side in lasers:
        for g in side:
            g["pts"].sort(key=lambda p: p[0])
        side.sort(key=lambda g: g["pts"][0][0])

    return {"meta": meta, "bpms": bpms, "sigs": sigs, "bt": bt, "fx": fx, "lasers": lasers}


def end_tick(chart: dict) -> int:
    end = 0
    for lane in chart["bt"]:
        for y, l in lane:
            end = max(end, y + max(l, 0))
    for side in chart["fx"]:
        for y, l, _ in side:
            end = max(end, y + max(l, 0))
    for side in chart["lasers"]:
        for g in side:
            if g["pts"]:
                end = max(end, g["pts"][-1][0])
    for y, _ in chart["bpms"]:
        end = max(end, y)
    return end


class Timing:
    """tick <-> ms mapping; chart tick 0 lies at audio time meta.o (ms)."""

    def __init__(self, chart: dict):
        try:
            o = float(chart["meta"].get("o", "0") or 0)
        except ValueError:
            o = 0.0
        self.segs: list[tuple[int, float, float]] = []  # (y, ms, bpm)
        ms, last_y, last_b = o, 0, chart["bpms"][0][1]
        for y, v in chart["bpms"]:
            ms += (y - last_y) * (60000.0 / (last_b * TICKS_PER_BEAT))
            self.segs.append((y, ms, v))
            last_y, last_b = y, v

    def tick_to_ms(self, t: float) -> float:
        lo, hi = 0, len(self.segs) - 1
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if self.segs[mid][0] <= t:
                lo = mid
            else:
                hi = mid - 1
        y, ms, bpm = self.segs[lo]
        return ms + (t - y) * (60000.0 / (bpm * TICKS_PER_BEAT))
