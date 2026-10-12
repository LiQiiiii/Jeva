from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from .backbone import TextBackbone
from .matrix_decision import MatrixDecisionStack


class System1DecisionModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        if model_cfg.decision_arch != "matrix_stack":
            raise ValueError("Jeva requires decision_arch=matrix_stack")
        self.backbone = TextBackbone(model_cfg)
        d = int(model_cfg.projection_dim)
        self.context_proj = nn.Linear(self.backbone.hidden_size, d, bias=False)
        self.option_proj = nn.Linear(self.backbone.hidden_size, d, bias=False)
        self.core = MatrixDecisionStack(
            d, depth=model_cfg.decision_depth, num_heads=model_cfg.decision_num_heads,
            ffn_ratio=model_cfg.decision_ffn_ratio, tau_set_init=model_cfg.tau_set_init,
            tau_dec_init=model_cfg.tau_dec_init, tau_cov_init=model_cfg.tau_cov_init,
            alpha_over_tau_init=model_cfg.alpha_over_tau_init,
            coverage_bias_init=model_cfg.coverage_bias_init,
            min_temperature=model_cfg.min_temperature, dropout=model_cfg.decision_dropout,
            zero_init_ffn=model_cfg.decision_zero_init_ffn,
            normalize_initial_state=model_cfg.normalize_state,
            final_norm=model_cfg.decision_final_norm,
        )
        if model_cfg.head_path:
            self.load_head(model_cfg.head_path)

    @property
    def tokenizer(self):
        return self.backbone.tokenizer

    def encode_state_inputs(self, inputs):
        h = self.backbone.encode(inputs)
        return self.context_proj(h.to(dtype=self.context_proj.weight.dtype))

    def encode_option_inputs(self, inputs, option_chunk_size=None):
        n = inputs["input_ids"].shape[0]
        chunk = option_chunk_size if option_chunk_size and option_chunk_size > 0 else n
        result = [self.backbone.encode({k: v[i:i + chunk] for k, v in inputs.items()})
                  for i in range(0, n, chunk)]
        h = torch.cat(result, dim=0)
        return self.option_proj(h.to(dtype=self.option_proj.weight.dtype))

    @staticmethod
    def _pack(flat, counts, kmax):
        out = flat.new_zeros((len(counts), kmax, flat.shape[-1]))
        offset = 0
        for b, k in enumerate(counts.tolist()):
            out[b, :k] = flat[offset:offset + k]
            offset += k
        return out

    def forward(self, batch, option_chunk_size=None):
        c = self.encode_state_inputs(batch["state_inputs"])
        flat = self.encode_option_inputs(batch["option_inputs"], option_chunk_size)
        u = self._pack(flat, batch["option_counts"], batch["option_mask"].shape[1])
        return self.decide_from_embeddings(c, u, batch["option_mask"])

    def decide_from_embeddings(self, c, u, option_mask):
        return self.core(c, u, option_mask)

    def prepare_option_kv_cache(self, u, option_mask, *, detach=True):
        return self.core.prepare_option_cache(u, option_mask, detach=detach)

    def decide_from_option_cache(self, c, cache):
        return self.core.forward_from_cache(c, cache)

    def compile_option_memory(self, u, *, detach=True):
        return self.core.compile_option_memory(u, detach=detach)

    def assemble_option_memory(self, memories):
        return self.core.assemble_option_memory(memories)

    def head_state_dict(self):
        return {"decision_arch": "matrix_stack", "context_proj": self.context_proj.state_dict(),
                "option_proj": self.option_proj.state_dict(), "core": self.core.state_dict()}

    def save_head(self, path: str | Path):
        torch.save(self.head_state_dict(), path)

    def load_head(self, path: str | Path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("decision_arch", "matrix_stack") != "matrix_stack":
            raise ValueError("head architecture does not match matrix_stack")
        for key in ("context_proj", "option_proj", "core"):
            getattr(self, key).load_state_dict(state[key], strict=True)
