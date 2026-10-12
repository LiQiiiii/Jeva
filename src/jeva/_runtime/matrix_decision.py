from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .layers import PositiveScalar, SwiGLU


@dataclass(frozen=True)
class MatrixOptionKVCache:


    option_read: torch.Tensor
    option_mask: torch.Tensor
    layer_keys: tuple[torch.Tensor, ...]
    layer_values: tuple[torch.Tensor, ...]

    @property
    def batch_size(self) -> int:
        return int(self.option_mask.shape[0])

    @property
    def num_options(self) -> int:
        return int(self.option_mask.shape[1])

    @property
    def nbytes(self) -> int:
        tensors = (self.option_read, self.option_mask, *self.layer_keys, *self.layer_values)
        return sum(t.numel() * t.element_size() for t in tensors)


@dataclass(frozen=True)
class FactorizedOptionMemory:


    option_read: torch.Tensor
    layer_keys: tuple[torch.Tensor, ...]
    layer_values: tuple[torch.Tensor, ...]

    @property
    def nbytes(self) -> int:
        tensors = (self.option_read, *self.layer_keys, *self.layer_values)
        return int(sum(t.numel() * t.element_size() for t in tensors))


class MatrixDecisionBlock(nn.Module):


    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        tau_set_init: float = 0.15,
        alpha_over_tau_init: float = 0.60,
        min_temperature: float = 1e-4,
        dropout: float = 0.0,
        zero_init_ffn: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if num_heads < 1 or dim % num_heads != 0:
            raise ValueError("num_heads must divide dim")

        self.dim = int(dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.depth_scale = 1.0 / math.sqrt(float(depth))


        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.o_proj = nn.Linear(self.dim, self.dim, bias=False)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.eye_(projection.weight)

        self.tau_set = PositiveScalar(tau_set_init, min_temperature)
        ratio = min(max(float(alpha_over_tau_init), 1e-4), 1.0 - 1e-4)
        self.gamma = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))

        self.ffn_ratio = float(ffn_ratio)
        if self.ffn_ratio > 0:
            ffn_dim = int(round(self.dim * self.ffn_ratio))
            self.ffn_norm = nn.RMSNorm(self.dim, eps=eps)
            self.ffn = SwiGLU(
                self.dim,
                ffn_dim,
                dropout=dropout,
                zero_init_output=zero_init_ffn,
            )
        else:
            self.ffn_norm = None
            self.ffn = None

    def alpha(self) -> torch.Tensor:
        return self.tau_set() * torch.sigmoid(self.gamma) * self.depth_scale

    def project_options(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        if u.ndim != 3 or u.shape[-1] != self.dim:
            raise ValueError(f"u must be [B,K,{self.dim}], got {tuple(u.shape)}")
        batch, options, _ = u.shape
        keys = self.k_proj(u).reshape(
            batch, options, self.num_heads, self.head_dim
        )
        values = self.v_proj(u).reshape(
            batch, options, self.num_heads, self.head_dim
        )


        keys = F.normalize(keys.float(), dim=-1)
        return keys, values

    def forward_from_kv(
        self,
        z: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        option_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        batch = z.shape[0]
        queries = self.q_proj(z).reshape(batch, self.num_heads, self.head_dim)
        queries = F.normalize(queries.float(), dim=-1)

        logits = torch.einsum("bhd,bkhd->bhk", queries, keys.float())
        tau = self.tau_set().float()
        set_logits = (logits / tau).masked_fill(
            ~option_mask.unsqueeze(1),
            float("-inf"),
        )
        attention = F.softmax(set_logits, dim=-1)
        attention = torch.where(
            option_mask.unsqueeze(1), attention, torch.zeros_like(attention)
        )

        message = torch.einsum(
            "bhk,bkhd->bhd", attention, values.float()
        ).reshape(batch, self.dim)
        message = self.o_proj(
            message.to(dtype=self.o_proj.weight.dtype)
        ).to(dtype=z.dtype)
        z = z + self.alpha().to(dtype=z.dtype) * message

        if self.ffn is not None:
            normed = self.ffn_norm(z.to(dtype=self.ffn_norm.weight.dtype))
            delta = self.ffn(normed).to(dtype=z.dtype)
            z = z + self.depth_scale * delta

        if return_attention:
            return z, attention, self.alpha()
        return z

    def forward(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        option_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        keys, values = self.project_options(u)
        return self.forward_from_kv(
            z,
            keys,
            values,
            option_mask,
            return_attention=return_attention,
        )


class MatrixDecisionStack(nn.Module):


    def __init__(
        self,
        dim: int,
        *,
        depth: int = 8,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        tau_set_init: float = 0.15,
        tau_dec_init: float = 0.10,
        tau_cov_init: float = 0.10,
        alpha_over_tau_init: float = 0.60,
        coverage_bias_init: float = 0.0,
        min_temperature: float = 1e-4,
        dropout: float = 0.0,
        zero_init_ffn: bool = True,
        normalize_initial_state: bool = False,
        final_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if num_heads < 1 or dim % num_heads != 0:
            raise ValueError("num_heads must divide dim")

        self.dim = int(dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.normalize_initial_state = bool(normalize_initial_state)

        self.blocks = nn.ModuleList([
            MatrixDecisionBlock(
                self.dim,
                depth=self.depth,
                num_heads=self.num_heads,
                ffn_ratio=ffn_ratio,
                tau_set_init=tau_set_init,
                alpha_over_tau_init=alpha_over_tau_init,
                min_temperature=min_temperature,
                dropout=dropout,
                zero_init_ffn=zero_init_ffn,
                eps=eps,
            )
            for _ in range(self.depth)
        ])

        self.final_norm = nn.RMSNorm(self.dim, eps=eps) if final_norm else nn.Identity()
        self.tau_dec = PositiveScalar(tau_dec_init, min_temperature)
        self.tau_cov = PositiveScalar(tau_cov_init, min_temperature)
        self.coverage_bias = nn.Parameter(torch.tensor(float(coverage_bias_init)))

    def _validate_options(
        self, u: torch.Tensor, option_mask: torch.Tensor
    ) -> None:
        if u.ndim != 3 or u.shape[-1] != self.dim:
            raise ValueError(f"u must be [B,K,{self.dim}], got {tuple(u.shape)}")
        if option_mask.ndim != 2 or option_mask.shape != u.shape[:2]:
            raise ValueError("option_mask must match u's [B,K] dimensions")
        if not torch.all(option_mask.any(dim=-1)):
            raise ValueError("every example must contain at least one valid option")

    def prepare_option_cache(
        self,
        u: torch.Tensor,
        option_mask: torch.Tensor,
        *,
        detach: bool = False,
    ) -> MatrixOptionKVCache:

        self._validate_options(u, option_mask)
        if detach and self.training:
            raise RuntimeError("detached option KV caches are inference-only")

        option_read = F.normalize(u, dim=-1)
        layer_kv = [block.project_options(option_read) for block in self.blocks]
        layer_keys = tuple(item[0] for item in layer_kv)
        layer_values = tuple(item[1] for item in layer_kv)

        if detach:
            option_read = option_read.detach()
            option_mask = option_mask.detach()
            layer_keys = tuple(value.detach() for value in layer_keys)
            layer_values = tuple(value.detach() for value in layer_values)

        return MatrixOptionKVCache(
            option_read=option_read,
            option_mask=option_mask,
            layer_keys=layer_keys,
            layer_values=layer_values,
        )

    def compile_option_memory(
        self,
        u: torch.Tensor,
        *,
        detach: bool = True,
    ) -> tuple[FactorizedOptionMemory, ...]:

        if u.ndim != 2 or u.shape[-1] != self.dim:
            raise ValueError(f"u must be [N,{self.dim}], got {tuple(u.shape)}")
        if u.shape[0] < 1:
            raise ValueError("at least one option is required")
        option_mask = torch.ones((1, u.shape[0]), dtype=torch.bool, device=u.device)
        cache = self.prepare_option_cache(
            u.unsqueeze(0), option_mask, detach=detach
        )


        memories = []
        for index in range(u.shape[0]):
            memories.append(
                FactorizedOptionMemory(
                    option_read=cache.option_read[0, index].clone(),
                    layer_keys=tuple(
                        keys[0, index].clone() for keys in cache.layer_keys
                    ),
                    layer_values=tuple(
                        values[0, index].clone() for values in cache.layer_values
                    ),
                )
            )
        return tuple(memories)

    def assemble_option_memory(
        self,
        memories: list[FactorizedOptionMemory]
        | tuple[FactorizedOptionMemory, ...],
    ) -> MatrixOptionKVCache:

        if not memories:
            raise ValueError("at least one option memory is required")
        for memory in memories:
            if len(memory.layer_keys) != self.depth or len(memory.layer_values) != self.depth:
                raise ValueError("option memory depth does not match model depth")
            if memory.option_read.shape != (self.dim,):
                raise ValueError("option memory dimension does not match model dimension")

        option_read = torch.stack(
            [memory.option_read for memory in memories], dim=0
        ).unsqueeze(0)
        layer_keys = tuple(
            torch.stack(
                [memory.layer_keys[layer] for memory in memories], dim=0
            ).unsqueeze(0)
            for layer in range(self.depth)
        )
        layer_values = tuple(
            torch.stack(
                [memory.layer_values[layer] for memory in memories], dim=0
            ).unsqueeze(0)
            for layer in range(self.depth)
        )
        option_mask = torch.ones(
            (1, len(memories)), dtype=torch.bool, device=option_read.device
        )
        return MatrixOptionKVCache(
            option_read=option_read,
            option_mask=option_mask,
            layer_keys=layer_keys,
            layer_values=layer_values,
        )

    def forward_from_cache(
        self,
        c: torch.Tensor,
        cache: MatrixOptionKVCache,
        *,
        return_layer_attentions: bool = False,
    ) -> dict[str, torch.Tensor]:
        if c.ndim != 2 or c.shape[-1] != self.dim:
            raise ValueError(f"c must be [B,{self.dim}], got {tuple(c.shape)}")
        if c.shape[0] != cache.batch_size:
            raise ValueError("state and option cache batch sizes differ")
        if len(cache.layer_keys) != self.depth or len(cache.layer_values) != self.depth:
            raise ValueError("option cache depth does not match model depth")

        z = F.normalize(c, dim=-1) if self.normalize_initial_state else c
        layer_attentions = []
        layer_alphas = []

        for index, block in enumerate(self.blocks):
            if return_layer_attentions:
                z, attention, alpha = block.forward_from_kv(
                    z,
                    cache.layer_keys[index],
                    cache.layer_values[index],
                    cache.option_mask,
                    return_attention=True,
                )
                layer_attentions.append(attention)
                layer_alphas.append(alpha)
            else:
                z = block.forward_from_kv(
                    z,
                    cache.layer_keys[index],
                    cache.layer_values[index],
                    cache.option_mask,
                )

        z_read = self.final_norm(z)
        scores = torch.einsum(
            "bd,bkd->bk", z_read.float(), cache.option_read.float()
        )
        masked_scores = scores.masked_fill(~cache.option_mask, float("-inf"))

        tau_dec = self.tau_dec().float()
        decision_logits = (scores / tau_dec).masked_fill(
            ~cache.option_mask,
            float("-inf"),
        )
        log_q = F.log_softmax(decision_logits, dim=-1)
        q = torch.where(cache.option_mask, log_q.exp(), torch.zeros_like(log_q))

        max_score = masked_scores.max(dim=-1).values
        tau_cov = self.tau_cov().float()
        coverage_logit = (max_score - self.coverage_bias.float()) / tau_cov
        coverage = torch.sigmoid(coverage_logit)
        p_null = torch.sigmoid(-coverage_logit)
        p_candidates = coverage.unsqueeze(-1) * q

        out = {
            "q": q,
            "coverage": coverage,
            "p_null": p_null,
            "p_candidates": p_candidates,
            "scores": scores,
            "z": z,
            "z_read": z_read,
            "tau_dec": self.tau_dec(),
            "tau_cov": self.tau_cov(),
            "decision_logits": decision_logits,
            "log_q": log_q,
            "coverage_logit": coverage_logit,
        }
        if return_layer_attentions:
            out["layer_attentions"] = layer_attentions
            out["layer_alphas"] = layer_alphas
        return out

    def forward(
        self,
        c: torch.Tensor,
        u: torch.Tensor,
        option_mask: torch.Tensor,
        *,
        return_layer_attentions: bool = False,
    ) -> dict[str, torch.Tensor]:
        cache = self.prepare_option_cache(u, option_mask, detach=False)
        return self.forward_from_cache(
            c,
            cache,
            return_layer_attentions=return_layer_attentions,
        )

    def logging_stats(self) -> dict[str, float]:
        taus = torch.stack([block.tau_set() for block in self.blocks])
        alphas = torch.stack([block.alpha() for block in self.blocks])
        return {
            "decision_depth": float(self.depth),
            "decision_num_heads": float(self.num_heads),
            "tau_set_mean": float(taus.mean().detach().cpu()),
            "tau_set_min": float(taus.min().detach().cpu()),
            "tau_set_max": float(taus.max().detach().cpu()),
            "alpha_mean": float(alphas.mean().detach().cpu()),
            "tau_dec": float(self.tau_dec().detach().cpu()),
            "tau_cov": float(self.tau_cov().detach().cpu()),
        }

    @staticmethod
    def approximate_stack_parameters(
        dim: int, depth: int, ffn_ratio: float = 4.0
    ) -> int:
        ffn_dim = int(round(dim * ffn_ratio))
        return int(depth * (4 * dim * dim + 3 * dim * ffn_dim))

    @staticmethod
    def approximate_stack_macs(
        dim: int, depth: int, k: int, ffn_ratio: float = 4.0
    ) -> int:
        ffn_dim = int(round(dim * ffn_ratio))

        return int(
            depth
            * (2 * dim * dim + 2 * k * dim * dim + 2 * k * dim + 3 * dim * ffn_dim)
        )
