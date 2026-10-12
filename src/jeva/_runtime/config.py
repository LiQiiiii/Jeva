from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml


@dataclass
class LoRAConfig:
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] | str = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class ModelConfig:
    backbone: str = "Qwen/Qwen3.5-4B"
    tokenizer_path: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    tune_mode: str = "lora"
    pooling: str = "last"
    projection_dim: int = 768
    normalize_state: bool = False
    decision_arch: str = "matrix_stack"
    decision_depth: int = 1
    decision_num_heads: int = 8
    decision_ffn_ratio: float = 4.0
    decision_dropout: float = 0.0
    decision_zero_init_ffn: bool = True
    decision_final_norm: bool = False
    gradient_checkpointing: bool = False
    torch_dtype: str = "bfloat16"
    tau_set_init: float = 0.15
    tau_dec_init: float = 0.10
    tau_cov_init: float = 0.10
    alpha_over_tau_init: float = 0.60
    coverage_bias_init: float = 0.0
    min_temperature: float = 1e-4
    head_path: str | None = None
    adapter_path: str | None = None
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class DataConfig:
    max_state_length: int = 9000
    max_option_length: int = 8192


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError("config.yaml must contain a model mapping")
    model = dict(raw["model"])
    lora = LoRAConfig(**(model.pop("lora", None) or {}))
    cfg = Config(model=ModelConfig(**model, lora=lora))
    data = raw.get("data", {}) or {}
    cfg.data = DataConfig(**{k: data[k] for k in ("max_state_length", "max_option_length") if k in data})
    if cfg.model.decision_arch != "matrix_stack":
        raise ValueError("Jeva requires decision_arch=matrix_stack")
    if cfg.model.torch_dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("torch_dtype must be float32, float16, or bfloat16")
    if cfg.model.pooling not in {"last", "mean", "cls"}:
        raise ValueError("pooling must be last, mean, or cls")
    if cfg.model.tune_mode not in {"lora", "frozen", "full"}:
        raise ValueError("tune_mode must be lora, frozen, or full")
    return cfg
