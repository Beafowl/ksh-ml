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
    audio_dim: int = 16  # onset-strength lookahead window (1/16 cells); 0 = no audio


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

    def forward(self, x):
        b, t, c = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(c, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x


class ChartGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.ctx, cfg.n_embd))
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

    def forward(self, idx, targets=None, audio=None):
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :t]
        if audio is not None and self.audio_proj is not None:
            x = x + self.audio_proj(audio)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
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
