"""Log-mel spectrogram features per 1/16-note grid cell (dataset v3).

The in-editor JS implementation mirrors this file exactly — any change here
must be ported to gen.js.

Definition (deterministic, no librosa):
  stft:      n_fft 1024, hop 512, hann window (same frames as onsets.py)
  mel:       N_MELS=64 triangular filters on the HTK mel scale over
             [0, sr/2]; weights from bin-center frequencies, un-normalized
  power:     mel_power = filt @ |stft|^2
  dB:        mel_db = 10*log10(mel_power + 1e-10)
  normalize: x = clip((mel_db - p95(mel_db)) / 60 + 1, 0, 1)   (per chart)
  cells:     per 1/16 cell, element-wise max over the cell frame +-1
  storage:   uint8, round(x * 255), shape (cells, 64)
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

N_FFT = 1024
HOP = 512
N_MELS = 64


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: float) -> np.ndarray:
    """(N_MELS, n_fft//2+1) triangular filters from bin-center frequencies."""
    n_bins = N_FFT // 2 + 1
    bin_hz = np.arange(n_bins) * sr / N_FFT
    pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2.0), N_MELS + 2))
    filt = np.zeros((N_MELS, n_bins), dtype=np.float32)
    for m in range(N_MELS):
        lo, mid, hi = pts[m], pts[m + 1], pts[m + 2]
        up = (bin_hz - lo) / max(mid - lo, 1e-9)
        down = (hi - bin_hz) / max(hi - mid, 1e-9)
        filt[m] = np.clip(np.minimum(up, down), 0.0, None)
    return filt


def mel_frames(path: str):
    """-> (mel_db ndarray (frames, N_MELS) float32, frames-per-second)"""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if len(mono) < N_FFT * 2:
        return np.full((1, N_MELS), -100.0, dtype=np.float32), sr / HOP
    n_frames = 1 + (len(mono) - N_FFT) // HOP
    window = np.hanning(N_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.as_strided(
        mono, shape=(n_frames, N_FFT),
        strides=(mono.strides[0] * HOP, mono.strides[0]))
    filt = mel_filterbank(sr).T  # (bins, mels)
    out = np.zeros((n_frames, N_MELS), dtype=np.float32)
    CH = 2048
    for a in range(0, n_frames, CH):
        b = min(n_frames, a + CH)
        power = np.abs(np.fft.rfft(frames[a:b] * window, axis=1)) ** 2
        out[a:b] = power @ filt
    return (10.0 * np.log10(out + 1e-10)).astype(np.float32), sr / HOP


def grid_mel_u8(mel_db: np.ndarray, fps: float, grid_ms: np.ndarray) -> np.ndarray:
    """-> uint8 (len(grid_ms), N_MELS), normalized per chart as documented."""
    ref = np.percentile(mel_db, 95)
    x = np.clip((mel_db - ref) / 60.0 + 1.0, 0.0, 1.0)
    idx = np.round(np.asarray(grid_ms, dtype=np.float64) / 1000.0 * fps).astype(np.int64)
    idx = np.clip(idx, 0, len(x) - 1)
    lo = np.clip(idx - 1, 0, len(x) - 1)
    hi = np.clip(idx + 1, 0, len(x) - 1)
    cells = np.maximum(np.maximum(x[lo], x[idx]), x[hi])
    return np.round(cells * 255.0).astype(np.uint8)
