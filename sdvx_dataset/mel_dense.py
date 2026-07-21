"""Dense per-frame log-mel for the v6 audio encoder (Mapperatorinator-style).

Fixed, tempo-independent frame rate: audio is resampled to 16 kHz and a
log-mel is taken every 10 ms (Whisper's config: n_fft 400, hop 160, 80 mels).
The in-editor JS mirrors this EXACTLY (OfflineAudioContext resample to 16 kHz
+ the same filterbank), like the v5 per-cell mel parity.

  resample:  -> 16000 Hz mono
  stft:      n_fft 400, hop 160, hann
  mel:       80 HTK triangular filters over [0, 8000]
  dB:        10*log10(power + 1e-10)
  normalize: clip((mel_db - p95)/60 + 1, 0, 1)   per song
  storage:   uint8 round(x*255), shape (frames, 80)   ~100 frames/sec
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
FPS = SR / HOP  # 100.0


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    n_bins = N_FFT // 2 + 1
    bin_hz = np.arange(n_bins) * SR / N_FFT
    pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(SR / 2.0), N_MELS + 2))
    filt = np.zeros((N_MELS, n_bins), dtype=np.float32)
    for m in range(N_MELS):
        lo, mid, hi = pts[m], pts[m + 1], pts[m + 2]
        up = (bin_hz - lo) / max(mid - lo, 1e-9)
        down = (hi - bin_hz) / max(hi - mid, 1e-9)
        filt[m] = np.clip(np.minimum(up, down), 0.0, None)
    return filt


def _resample_to_16k(mono: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return mono
    from math import gcd
    g = gcd(int(sr), SR)
    up, down = SR // g, int(sr) // g
    from scipy.signal import resample_poly
    return resample_poly(mono, up, down).astype(np.float32)


def dense_mel_u8(path: str, end_ms: float | None = None) -> np.ndarray:
    """-> uint8 (frames, 80). If end_ms given, trims to that span + 2 s."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    mono = _resample_to_16k(mono, sr)
    if end_ms is not None:
        keep = int((end_ms / 1000.0 + 2.0) * SR)
        mono = mono[:max(keep, N_FFT * 2)]
    if len(mono) < N_FFT * 2:
        return np.zeros((1, N_MELS), dtype=np.uint8)
    n_frames = 1 + (len(mono) - N_FFT) // HOP
    window = np.hanning(N_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.as_strided(
        mono, shape=(n_frames, N_FFT),
        strides=(mono.strides[0] * HOP, mono.strides[0]))
    filt = mel_filterbank().T  # (bins, mels)
    out = np.zeros((n_frames, N_MELS), dtype=np.float32)
    CH = 4096
    for a in range(0, n_frames, CH):
        b = min(n_frames, a + CH)
        power = np.abs(np.fft.rfft(frames[a:b] * window, axis=1)) ** 2
        out[a:b] = power @ filt
    mel_db = 10.0 * np.log10(out + 1e-10)
    ref = np.percentile(mel_db, 95)
    x = np.clip((mel_db - ref) / 60.0 + 1.0, 0.0, 1.0)
    return np.round(x * 255.0).astype(np.uint8)
