"""Baseline temporal ranking model: small CNN encoder + GRU + scoring head.

``temporal_ranker_v1`` (techspec section 16.3/16.4): the baseline from the
benchmark list is CNN + GRU — the smallest architecture that sees both
appearance (per-crop CNN) and temporal behavior (GRU over the frame
embeddings). Transformer variants are the section-15 comparison, not the
baseline. With single-digit positive examples, parameter count is kept
deliberately small (~60k).

Input:  crops ``[B, T, 1, K, K]`` + true lengths ``[B]`` (zero-padded
        tails are masked via packed sequences).
Output: one logit per track (sigmoid -> ranking score in [0, 1]).
"""

from __future__ import annotations

import torch
from torch import nn

MODEL_VERSION = "temporal_ranker_v1"


class CropEncoder(nn.Module):
    """Per-frame crop -> embedding. Three conv blocks + global average pool."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, embed_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalRanker(nn.Module):
    """CNN + GRU + linear scoring head over a padded crop stack.

    With ``dna_dim > 0`` a per-track Motion DNA feature vector is
    concatenated to the GRU hidden state before the scoring head
    (tracker section 15 experiment; the crop-only baseline is dna_dim=0).
    """

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 64,
                 dna_dim: int = 0):
        super().__init__()
        self.dna_dim = dna_dim
        self.encoder = CropEncoder(embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim + dna_dim, 1)

    def forward(self, crops: torch.Tensor, lengths: torch.Tensor,
                dna: torch.Tensor | None = None) -> torch.Tensor:
        batch, frames = crops.shape[:2]
        embeddings = self.encoder(crops.flatten(0, 1)).view(batch, frames, -1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)  # hidden: [1, B, H] = last real frame
        features = hidden.squeeze(0)
        if self.dna_dim:
            if dna is None:
                raise ValueError("model built with dna_dim > 0 but no dna "
                                 "features were passed")
            features = torch.cat([features, dna], dim=-1)
        return self.head(features).squeeze(-1)
