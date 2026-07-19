"""Minimal decoder-only transformer (GPT) for chart token sequences."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    ctx: int = 2048
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    audio_dim: int = 16   # flattened per-token audio input; 0 = no audio
    mel_bins: int = 0     # >0: input is (window x mel_bins) and a learned
                          # per-cell encoder replaces the flat projection
    # v5: bidirectional audio encoder + windowed cross-attention read.
    # enc_layer > 0 switches to (mel, cells) conditioning; audio_dim ignored.
    enc_layer: int = 0
    enc_embd: int = 256
    enc_head: int = 4
    cross_window: int = 16  # memory cells each token reads
    cross_back: int = 4     # of which this many lie behind the token
    max_cells: int = 4096   # encoder position-embedding capacity


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
        )
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout

    def forward(self, x, past_kv=None, attn_mask=None):
        """past_kv: optional (k, v) each (B, nh, P, hd) for incremental decoding.
        attn_mask: optional additive float mask (B or 1, 1, T, P+T); when
        neither is given the standard causal mask applies. Returns
        (x, (k, v)) with the mask-free training path unchanged."""
        b, t, c = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(c, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        if attn_mask is not None:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            a = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x, (k, v)


class EncBlock(nn.Module):
    """Bidirectional pre-LN transformer block for the audio encoder."""

    def __init__(self, d: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d, bias=False),
            nn.GELU(),
            nn.Linear(4 * d, d, bias=False),
        )
        self.n_head = n_head
        self.dropout = dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(c, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(b, t, c))
        x = x + self.mlp(self.ln2(x))
        return x


class AudioEncoder(nn.Module):
    """Whole-song mel cells (B,N,mel_bins) -> contextual memory (B,N,enc_embd)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.inp = nn.Linear(cfg.mel_bins, cfg.enc_embd)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_cells, cfg.enc_embd))
        self.blocks = nn.ModuleList(
            EncBlock(cfg.enc_embd, cfg.enc_head, cfg.dropout)
            for _ in range(cfg.enc_layer))
        self.ln = nn.LayerNorm(cfg.enc_embd)

    def forward(self, mel):
        x = self.inp(mel) + self.pos[:, :mel.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.ln(x)


class CrossRead(nn.Module):
    """Per-token attention read over a window of encoder memory around the
    token's grid cell. Query comes from the token embedding; a learned
    relative-cell embedding is added to the gathered memory. With zeroed
    memory (audio-free rows) the read degrades to a learned null vector,
    identically in training and inference."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.W, self.back, self.nh = cfg.cross_window, cfg.cross_back, cfg.n_head
        self.ln_q = nn.LayerNorm(cfg.n_embd)
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.kv = nn.Linear(cfg.enc_embd, 2 * cfg.n_embd, bias=False)
        self.rel = nn.Parameter(torch.zeros(cfg.cross_window, cfg.enc_embd))
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, memory, cells):
        # x (B,S,C), memory (B,N,E), cells (B,S) int64
        B, S, C = x.shape
        N, E = memory.shape[1], memory.shape[2]
        idx = cells[:, :, None] - self.back + torch.arange(
            self.W, device=cells.device)[None, None, :]
        idx = idx.clamp(0, N - 1).reshape(B, S * self.W, 1).expand(-1, -1, E)
        mem = memory.gather(1, idx).reshape(B, S, self.W, E) + self.rel
        k, v = self.kv(mem).split(C, dim=-1)
        hd = C // self.nh
        q = self.q(self.ln_q(x)).view(B, S, self.nh, hd)
        k = k.view(B, S, self.W, self.nh, hd)
        v = v.view(B, S, self.W, self.nh, hd)
        att = torch.einsum("bshd,bswhd->bshw", q, k) / math.sqrt(hd)
        out = torch.einsum("bshw,bswhd->bshd", att.softmax(dim=-1), v)
        return self.proj(out.reshape(B, S, C))


class ChartGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.ctx, cfg.n_embd))
        if cfg.enc_layer:
            self.encoder = AudioEncoder(cfg)
            self.cross = CrossRead(cfg)
            self.audio_proj = None
        elif cfg.mel_bins:
            # learned audio encoder: shared per-cell MLP over mel bins, then a
            # projection over the flattened lookahead window
            cells = cfg.audio_dim // cfg.mel_bins
            self.audio_proj = nn.Sequential(
                nn.Unflatten(-1, (cells, cfg.mel_bins)),
                nn.Linear(cfg.mel_bins, 32),
                nn.GELU(),
                nn.Flatten(-2),
                nn.Linear(cells * 32, cfg.n_embd, bias=False),
            )
        else:
            self.audio_proj = nn.Linear(cfg.audio_dim, cfg.n_embd, bias=False) if cfg.audio_dim else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, audio=None, mel=None, cells=None,
                memory=None, audio_off=None):
        """v4 path: audio (B,S,audio_dim). v5 path (enc_layer>0): mel (B,N,bins)
        or precomputed memory (B,N,enc_embd), plus cells (B,S) grid indices;
        audio_off (B,) bool zeroes a row's memory (the audio-free CFG row)."""
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :t]
        if self.cfg.enc_layer and cells is not None:
            if memory is None:
                memory = self.encoder(mel)
            if audio_off is not None:
                memory = memory * (~audio_off).to(memory.dtype)[:, None, None]
            x = x + self.cross(x, memory, cells)
        elif audio is not None and self.audio_proj is not None:
            x = x + self.audio_proj(audio)
        x = self.drop(x)
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-100)
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_kv(self, idx, audio, attn_mask, past, memory=None, cells=None):
        """Incremental forward for export/inference. idx (B,S); audio (B,S,A)
        for v4 models, or memory (B,N,E) + cells (B,S) for v5 models;
        attn_mask additive (B,1,S,P+S); past: list of n_layer (k,v) tensors
        with P=0 allowed. Returns (logits, presents)."""
        b, s = idx.shape
        p = past[0][0].shape[2]
        x = self.tok_emb(idx) + self.pos_emb[:, p:p + s]
        if self.cfg.enc_layer and cells is not None:
            x = x + self.cross(x, memory, cells)
        elif audio is not None and self.audio_proj is not None:
            x = x + self.audio_proj(audio)
        presents = []
        for blk, pkv in zip(self.blocks, past):
            x, kv = blk(x, past_kv=pkv, attn_mask=attn_mask)
            presents.append(kv)
        x = self.ln_f(x)
        return self.head(x), presents

    @torch.no_grad()
    def generate_step(self, idx, temperature=1.0, top_p=0.95, guidance=None, audio=None):
        """One sampling step. idx: (B, T). With guidance, row 0 is conditional
        and row 1 unconditional; returns the next token id (int)."""
        idx = idx[:, -self.cfg.ctx:]
        if audio is not None:
            audio = audio[:, -self.cfg.ctx:]
        logits, _ = self(idx, audio=audio)
        logits = logits[:, -1, :]
        if guidance is not None and logits.size(0) == 2:
            logits = logits[1] + guidance * (logits[0] - logits[1])
        else:
            logits = logits[0]
        logits = logits / max(1e-6, temperature)
        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            keep = torch.cumsum(sp, 0) - sp < top_p
            keep[0] = True
            probs = torch.zeros_like(probs).scatter_(0, si[keep], sp[keep])
            probs /= probs.sum()
        return int(torch.multinomial(probs, 1).item())
