"""Match converted chart folders to music_db songs by normalized title.

Passes: manual alias -> exact normalized title -> kana-folded title ->
suffix-stripped ("sdvx edit" style) title. Title collisions are resolved by
BPM closeness.
"""
from __future__ import annotations

import re
import unicodedata


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return "".join(ch for ch in s if ch.isalnum())


def kana_fold(s: str) -> str:
    """hiragana -> katakana so わんだふる matches ワンダフル."""
    out = []
    for ch in s:
        o = ord(ch)
        out.append(chr(o + 0x60) if 0x3041 <= o <= 0x3096 else ch)
    return "".join(out)


_SUFFIXES = ("sdvxedit", "sdvxedit2", "edit", "sdvxversion", "sdvxver")


def strip_suffix(s: str) -> str:
    for suf in _SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf) + 2:
            return s[: -len(suf)]
    return s


class TitleIndex:
    def __init__(self, songs):
        # key -> list of songs (collisions kept for bpm tiebreak)
        self.exact: dict[str, list] = {}
        self.folded: dict[str, list] = {}
        self.stripped: dict[str, list] = {}
        self.by_id = {s.music_id: s for s in songs}
        for s in songs:
            k = norm(s.title)
            kf = kana_fold(k)
            ks = strip_suffix(kf)
            self.exact.setdefault(k, []).append(s)
            self.folded.setdefault(kf, []).append(s)
            self.stripped.setdefault(ks, []).append(s)

    def match(self, title: str, ksh_bpm: float | None, alias_id: int | None):
        """-> (Song|None, how)"""
        if alias_id is not None:
            return self.by_id.get(alias_id), "alias"
        k = norm(title)
        kf = kana_fold(k)
        for key, table, how in (
            (k, self.exact, "exact"),
            (kf, self.folded, "kana"),
            (strip_suffix(kf), self.stripped, "suffix"),
        ):
            cands = table.get(key)
            if cands:
                return self._pick(cands, ksh_bpm), how
        return None, "none"

    @staticmethod
    def _pick(cands, ksh_bpm):
        if len(cands) == 1 or ksh_bpm is None:
            return cands[0]
        return min(cands, key=lambda s: abs(s.bpm_max - ksh_bpm))


def ksh_max_bpm(t_field: str) -> float | None:
    """meta 't' is '150' or '75-300' -> max value."""
    nums = re.findall(r"[\d.]+", t_field or "")
    try:
        return max(float(n) for n in nums) if nums else None
    except ValueError:
        return None
