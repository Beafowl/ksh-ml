"""Dense per-frame log-mel for the v6 audio encoder.

Parity-first design (mirrors the v5 mel that hit byte-identical JS parity):
NO resampling — the audio's native sample rate is used directly, with a hop
of round(0.01*sr) samples so the frame rate is ~100 fps (exactly 100 for the
44.1/48 kHz sources here), and a power-of-two FFT the browser can compute.
The in-editor JS mirrors this EXACTLY (same n_fft 1024 / filterbank as v5,
just 80 bins and dense output).

  frames:    n_fft 1024, hop round(0.01*sr) (~10 ms), hann
  mel:       80 HTK triangular filters over [0, sr/2] (v5 filterbank)
  dB:        10*log10(power + 1e-10)
  normalize: clip((mel_db - p95)/60 + 1, 0, 1)   per song
  storage:   uint8 round(x*255), shape (frames, 80)   ~100 frames/sec
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

N_FFT = 1024
N_MELS = 80
FPS = 100.0  # nominal; = sr / round(0.01*sr), exact for sr multiple of 100


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: float) -> np.ndarray:
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


def hop_for(sr: int) -> int:
    return int(round(0.01 * sr))


def dense_mel_u8(path: str, end_ms: float | None = None) -> np.ndarray:
    """-> uint8 (frames, 80). If end_ms given, trims to that span + 2 s."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    hop = hop_for(sr)
    if end_ms is not None:
        keep = int((end_ms / 1000.0 + 2.0) * sr)
        mono = mono[:max(keep, N_FFT * 2)]
    if len(mono) < N_FFT * 2:
        return np.zeros((1, N_MELS), dtype=np.uint8)
    n_frames = 1 + (len(mono) - N_FFT) // hop
    window = np.hanning(N_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.as_strided(
        mono, shape=(n_frames, N_FFT),
        strides=(mono.strides[0] * hop, mono.strides[0]))
    filt = mel_filterbank(sr).T  # (bins, mels)
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
