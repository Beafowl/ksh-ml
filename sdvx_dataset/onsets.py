"""Audio features per 1/16-note grid cell.

v2: 4 features per cell instead of a single broadband onset value —
  [0] spectral flux, low band   (< 200 Hz: kick / bass)
  [1] spectral flux, mid band   (200-2000 Hz: snare / melody)
  [2] spectral flux, high band  (> 2000 Hz: hats / crash)
  [3] RMS loudness (log-scaled)
Each feature is normalized by its own 98th percentile and clipped to 0..1.
Flux says "something hits here"; RMS says "this section is loud/quiet" —
the model needs both to place chords on accents and thin out quiet parts.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

N_FFT = 1024
HOP = 512
BAND_EDGES_HZ = (200.0, 2000.0)
FEATS = 4


def onset_features(path: str):
    """-> (features ndarray (frames, FEATS), frames-per-second)"""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if len(mono) < N_FFT * 2:
        return np.zeros((1, FEATS), dtype=np.float32), sr / HOP

    n_frames = 1 + (len(mono) - N_FFT) // HOP
    window = np.hanning(N_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.as_strided(
        mono,
        shape=(n_frames, N_FFT),
        strides=(mono.strides[0] * HOP, mono.strides[0]),
    )
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    b1 = np.searchsorted(freqs, BAND_EDGES_HZ[0])
    b2 = np.searchsorted(freqs, BAND_EDGES_HZ[1])

    out = np.zeros((n_frames, FEATS), dtype=np.float32)
    prev = None
    CH = 2048
    for a in range(0, n_frames, CH):
        b = min(n_frames, a + CH)
        chunk = frames[a:b]
        mag = np.log1p(np.abs(np.fft.rfft(chunk * window, axis=1)))
        block = np.vstack([prev, mag]) if prev is not None else mag
        d = np.diff(block, axis=0)
        np.clip(d, 0, None, out=d)
        lo = a if prev is not None else a + 1
        rows = d if prev is not None else d[: b - a - 1]
        if len(rows):
            out[lo:lo + len(rows), 0] = rows[:, :b1].sum(axis=1)
            out[lo:lo + len(rows), 1] = rows[:, b1:b2].sum(axis=1)
            out[lo:lo + len(rows), 2] = rows[:, b2:].sum(axis=1)
        out[a:b, 3] = np.log1p(np.sqrt((chunk ** 2).mean(axis=1)) * 20.0)
        prev = mag[-1:]

    for f in range(FEATS):
        ref = np.percentile(out[:, f], 98)
        if ref > 0:
            out[:, f] = np.clip(out[:, f] / ref, 0.0, 1.0)
    return out, sr / HOP


def grid_features(feats: np.ndarray, fps: float, grid_ms: np.ndarray) -> np.ndarray:
    """Sample every feature at each grid time (max over +-1 frame) ->
    (len(grid_ms), FEATS)."""
    idx = np.round(np.asarray(grid_ms, dtype=np.float64) / 1000.0 * fps).astype(np.int64)
    idx = np.clip(idx, 0, len(feats) - 1)
    lo = np.clip(idx - 1, 0, len(feats) - 1)
    hi = np.clip(idx + 1, 0, len(feats) - 1)
    return np.maximum(np.maximum(feats[lo], feats[idx]), feats[hi])


# ---- v1 API kept for compatibility (scalar broadband onset) ----

def onset_envelope(path: str):
    feats, fps = onset_features(path)
    flux = feats[:, 0] + feats[:, 1] + feats[:, 2]
    ref = np.percentile(flux, 98)
    if ref > 0:
        flux = np.clip(flux / ref, 0.0, 1.0)
    return flux.astype(np.float32), fps


def grid_onsets(env: np.ndarray, fps: float, grid_ms: np.ndarray) -> np.ndarray:
    idx = np.round(np.asarray(grid_ms, dtype=np.float64) / 1000.0 * fps).astype(np.int64)
    idx = np.clip(idx, 0, len(env) - 1)
    lo = np.clip(idx - 1, 0, len(env) - 1)
    hi = np.clip(idx + 1, 0, len(env) - 1)
    return np.maximum(np.maximum(env[lo], env[idx]), env[hi])
