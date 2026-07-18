"""Onset-strength features: ogg -> log-spectral-flux envelope -> one value per
1/16-note grid cell of the chart (aligned through the chart's timing map)."""
from __future__ import annotations

import numpy as np
import soundfile as sf

N_FFT = 1024
HOP = 512


def onset_envelope(path: str):
    """-> (envelope ndarray, frames-per-second) — spectral flux, 0..1-ish."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if len(mono) < N_FFT * 2:
        return np.zeros(1, dtype=np.float32), sr / HOP

    n_frames = 1 + (len(mono) - N_FFT) // HOP
    window = np.hanning(N_FFT).astype(np.float32)
    # strided frame view -> windowed rFFT magnitudes, processed in chunks
    frames = np.lib.stride_tricks.as_strided(
        mono,
        shape=(n_frames, N_FFT),
        strides=(mono.strides[0] * HOP, mono.strides[0]),
    )
    flux = np.zeros(n_frames, dtype=np.float32)
    prev = None
    CH = 2048
    for a in range(0, n_frames, CH):
        b = min(n_frames, a + CH)
        mag = np.log1p(np.abs(np.fft.rfft(frames[a:b] * window, axis=1)))
        block = np.vstack([prev, mag]) if prev is not None else mag
        d = np.diff(block, axis=0)
        np.clip(d, 0, None, out=d)
        s = d.sum(axis=1)
        if prev is None:
            flux[a + 1:b] = s[: b - a - 1] if b - a > 1 else []
        else:
            flux[a:b] = s
        prev = mag[-1:]
    ref = np.percentile(flux, 98)
    if ref > 0:
        flux = np.clip(flux / ref, 0.0, 1.0)
    return flux.astype(np.float32), sr / HOP


def grid_onsets(env: np.ndarray, fps: float, grid_ms: np.ndarray) -> np.ndarray:
    """Sample the envelope at each grid time (max over +-1 frame)."""
    idx = np.round(grid_ms / 1000.0 * fps).astype(np.int64)
    idx = np.clip(idx, 0, len(env) - 1)
    lo = np.clip(idx - 1, 0, len(env) - 1)
    hi = np.clip(idx + 1, 0, len(env) - 1)
    return np.maximum(np.maximum(env[lo], env[idx]), env[hi])
