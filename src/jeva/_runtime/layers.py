from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class PositiveScalar(nn.Module):
    def __init__(self, init: float, minimum: float = 1e-4):
        super().__init__()
        self.minimum = float(minimum)
        value = max(float(init) - self.minimum, 1e-6)
        self.raw = nn.Parameter(torch.tensor(math.log(math.expm1(value))))

    def forward(self):
        return self.minimum + F.softplus(self.raw)


class SwiGLU(nn.Module):
    def __init__(self, dim, ffn_dim, *, dropout=0.0, zero_init_output=True):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        if zero_init_output:
            nn.init.zeros_(self.down_proj.weight)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
