"""Parse SDVX music_db.xml (Shift-JIS) into song entries with per-difficulty
levels (difnum/10, fractional above 17) and official radar values (0-200)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

RADAR_AXES = ["notes", "peak", "tsumami", "tricky", "hand-trip", "one-hand"]
SLOTS = ["novice", "advanced", "exhaust", "infinite", "maximum"]

_MUSIC_RE = re.compile(r'<music id="(\d+)">([\s\S]*?)</music>')
_SLOT_RE = re.compile(r"<(novice|advanced|exhaust|infinite|maximum)>([\s\S]*?)</\1>")
_RADAR_RE = re.compile(r"<radar>([\s\S]*?)</radar>")
_FIELD_RE = re.compile(r"<([\w-]+)[^>]*>([^<]*)</")


@dataclass
class Difficulty:
    slot: str
    level: float          # difnum / 10 (e.g. 18.1); 0 when absent
    radar: dict | None    # {axis: 0-200} or None
    effected_by: str = ""


@dataclass
class Song:
    music_id: int
    title: str
    ascii: str
    bpm_min: float
    bpm_max: float
    version: int
    diffs: dict = field(default_factory=dict)  # slot -> Difficulty


def _text_field(body: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>([^<]*)</{tag}>", body)
    return m.group(1) if m else ""


def load(path: str) -> list[Song]:
    text = open(path, "rb").read().decode("cp932", errors="replace")
    songs = []
    for m in _MUSIC_RE.finditer(text):
        music_id, body = int(m.group(1)), m.group(2)
        try:
            bpm_max = int(_text_field(body, "bpm_max") or 0) / 100.0
            bpm_min = int(_text_field(body, "bpm_min") or 0) / 100.0
        except ValueError:
            bpm_max = bpm_min = 0.0
        song = Song(
            music_id=music_id,
            title=_text_field(body, "title_name"),
            ascii=_text_field(body, "ascii"),
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            version=int(_text_field(body, "version") or 0),
        )
        for sm in _SLOT_RE.finditer(body):
            slot, sbody = sm.group(1), sm.group(2)
            difnum = int(_text_field(sbody, "difnum") or 0)
            if difnum <= 0:
                continue
            radar = None
            rm = _RADAR_RE.search(sbody)
            if rm:
                radar = {}
                for fm in _FIELD_RE.finditer(rm.group(1)):
                    if fm.group(1) in RADAR_AXES:
                        radar[fm.group(1)] = int(fm.group(2))
                if len(radar) != len(RADAR_AXES):
                    radar = None
            song.diffs[slot] = Difficulty(
                slot=slot,
                level=difnum / 10.0,
                radar=radar,
                effected_by=_text_field(sbody, "effected_by"),
            )
        if song.diffs:
            songs.append(song)
    return songs
