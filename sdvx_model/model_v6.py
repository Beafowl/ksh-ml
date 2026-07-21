"""v6 model: Whisper-style audio encoder-decoder for mania charts.

  dense mel (B, F, 80)                      event tokens (B, T)
        |                                          |
   Conv stem (x2 downsample)                 token + pos embed
        |                                          |
   + sinusoidal pos                          M x DecoderBlock:
        |                                       - causal self-attn (KV cache)
   N x EncoderBlock (bidir self-attn)  --->    - cross-attn to memory
        |                                       - FFN
   memory (B, F/2, D)  --------------------->  head -> logits (B, T, V)

Faithful to Mapperatorinator's seq2seq recipe, sized for a 3070 + browser
(~45M params). Cross-attention K/V depend only on the encoder memory, so at
inference they are computed once per window and cached alongside the causal
self-attention KV cache (the standard Whisper decoder export pattern).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfigV6:
    vocab_size: int
    mel_bins: int = 80
    d_model: int = 512
    n_head: int = 8
    d_ff: int = 2048
    enc_layer: int = 6
    dec_layer: int = 6
    max_audio: int = 1600     # encoder frames before conv downsample
    max_tgt: int = 1024       # decoder positions
    dropout: float = 0.1


def sinusoids(length, channels):
    inv = torch.exp(-math.log(10000) / (channels // 2 - 1) *
                    torch.arange(channels // 2))
    t = torch.arange(length)[:, None] * inv[None, :]
    return torch.cat([t.sin(), t.cos()], dim=1)


class MHA(nn.Module):
    """Multi-head attention supporting self/cross and KV caching."""

    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.n_head, self.dropout = n_head, dropout
        self.q = nn.Linear(d, d, bias=True)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=True)
        self.o = nn.Linear(d, d, bias=True)

    def _split(self, x, b, t):
        return x.view(b, t, self.n_head, -1).transpose(1, 2)

    def forward(self, x, ctx=None, causal=False, past_kv=None, cache_kv=None):
        """x: (B,T,D) query source. ctx: cross-attn keys source (else self).
        past_kv: (k,v) to prepend (self-attn incremental).
        cache_kv: precomputed (k,v) for cross-attn (skip k/v projection).
        Returns (out, (k,v)) where (k,v) is the full key/value used."""
        b, t, d = x.shape
        q = self._split(self.q(x), b, t)
        if cache_kv is not None:
            k, v = cache_kv
        else:
            src = ctx if ctx is not None else x
            k = self._split(self.k(src), b, src.shape[1])
            v = self._split(self.v(src), b, src.shape[1])
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal and past_kv is None and cache_kv is None,
            dropout_p=self.dropout if self.training else 0.0)
        a = a.transpose(1, 2).contiguous().view(b, t, d)
        return self.o(a), (k, v)


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ConfigV6):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MHA(cfg.d_model, cfg.n_head, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                                 nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ConfigV6):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = MHA(cfg.d_model, cfg.n_head, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = MHA(cfg.d_model, cfg.n_head, cfg.dropout)
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                                 nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x, memory, self_past=None, cross_cache=None):
        h, self_kv = self.self_attn(self.ln1(x), causal=True, past_kv=self_past)
        x = x + h
        h, cross_kv = self.cross_attn(self.ln2(x), ctx=memory, cache_kv=cross_cache)
        x = x + h
        x = x + self.mlp(self.ln3(x))
        return x, self_kv, cross_kv


class ChartV6(nn.Module):
    def __init__(self, cfg: ConfigV6):
        super().__init__()
        self.cfg = cfg
        # audio stem: conv1 keeps length, conv2 downsamples x2 (Whisper)
        self.conv1 = nn.Conv1d(cfg.mel_bins, cfg.d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(cfg.d_model, cfg.d_model, 3, stride=2, padding=1)
        self.register_buffer("enc_pos", sinusoids(cfg.max_audio // 2 + 1, cfg.d_model))
        self.enc = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.enc_layer))
        self.enc_ln = nn.LayerNorm(cfg.d_model)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dec_pos = nn.Parameter(torch.zeros(1, cfg.max_tgt, cfg.d_model))
        self.dec = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.dec_layer))
        self.dec_ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, mel):
        # mel (B, F, 80) -> memory (B, F/2, D)
        x = F.gelu(self.conv1(mel.transpose(1, 2)))
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        x = x + self.enc_pos[:x.shape[1]].to(x.dtype)
        for blk in self.enc:
            x = blk(x)
        return self.enc_ln(x)

    def forward(self, mel, idx, targets=None, memory=None):
        if memory is None:
            memory = self.encode(mel)
        b, t = idx.shape
        x = self.tok_emb(idx) + self.dec_pos[:, :t]
        for blk in self.dec:
            x, _, _ = blk(x, memory)
        logits = self.head(self.dec_ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # ---- inference helpers ----
    def cross_cache(self, memory):
        """Precompute cross-attn K/V for every decoder layer (once per window)."""
        cache = []
        for blk in self.dec:
            # cross_attn query is ln2(decoder_x); K/V come from raw memory
            k = blk.cross_attn._split(blk.cross_attn.k(memory), memory.shape[0], memory.shape[1])
            v = blk.cross_attn._split(blk.cross_attn.v(memory), memory.shape[0], memory.shape[1])
            cache.append((k, v))
        return cache

    def decode_step(self, idx, cross, self_past, pos):
        """One/few-token decode. idx (B,S); cross: per-layer (k,v);
        self_past: per-layer (k,v) or None; pos: start position for pos emb.
        Returns (logits, new_self_past)."""
        b, s = idx.shape
        x = self.tok_emb(idx) + self.dec_pos[:, pos:pos + s]
        new_past = []
        for i, blk in enumerate(self.dec):
            h, self_kv = blk.self_attn(blk.ln1(x), causal=True,
                                       past_kv=self_past[i] if self_past else None)
            x = x + h
            h, _ = blk.cross_attn(blk.ln2(x), cache_kv=cross[i])
            x = x + h
            x = x + blk.mlp(blk.ln3(x))
            new_past.append(self_kv)
        return self.head(self.dec_ln(x)), new_past
