from __future__ import annotations

from pathlib import Path
import warnings
import torch
from torch import nn

from transformers import AutoModel, AutoTokenizer


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _saved_adapter_keys(adapter_path: str) -> set[str]:

    safe_path = Path(adapter_path) / "adapter_model.safetensors"
    if safe_path.is_file():
        from safetensors import safe_open

        with safe_open(str(safe_path), framework="pt", device="cpu") as handle:
            return set(handle.keys())

    from peft.utils.save_and_load import load_peft_weights

    return set(load_peft_weights(adapter_path, device="cpu"))


def _lora_keys(keys) -> set[str]:
    return {key for key in keys if ".lora_" in key}


def _qwen35_adapter_layout(keys: set[str]) -> str:

    lora_keys = _lora_keys(keys)
    wrapped = "base_model.model.language_model."
    text_only = "base_model.model.layers."
    if lora_keys and all(key.startswith(wrapped) for key in lora_keys):
        return "qwen35_wrapper"
    if lora_keys and all(key.startswith(text_only) for key in lora_keys):
        return "qwen35_text_only"
    raise ValueError(
        "Unsupported or mixed Qwen3.5 LoRA key hierarchy; refusing a partial load. "
        f"Sample keys: {sorted(lora_keys or keys)[:3]}"
    )


def _load_lora_strict(model, adapter_path: str, saved_keys: set[str]):

    from peft import PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict

    expected = _lora_keys(saved_keys)
    if not expected:
        raise ValueError(f"No LoRA tensors found in {adapter_path}")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=r".*missing adapter keys.*")
            loaded = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    except UserWarning as exc:
        raise RuntimeError(
            f"Incomplete LoRA load from {adapter_path}; checkpoint/model hierarchies "
            f"must match. Refusing to train, merge or evaluate. {str(exc)[:600]}"
        ) from exc


    actual = _lora_keys(get_peft_model_state_dict(loaded, save_embedding_layers=False))
    if actual != expected:
        raise RuntimeError(
            f"LoRA key mismatch for {adapter_path}: "
            f"checkpoint-only={len(expected - actual)} {sorted(expected - actual)[:3]}, "
            f"model-only={len(actual - expected)} {sorted(actual - expected)[:3]}. "
            "Refusing a partial adapter load."
        )
    return loaded


class TextBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dtype = DTYPES.get(cfg.torch_dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_path or cfg.backbone,
            trust_remote_code=cfg.trust_remote_code,
            local_files_only=cfg.local_files_only,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "right"


        kwargs = dict(
            pretrained_model_name_or_path=cfg.backbone,
            trust_remote_code=cfg.trust_remote_code,
            local_files_only=cfg.local_files_only,
        )
        try:
            self.model = AutoModel.from_pretrained(dtype=dtype, **kwargs)
        except TypeError:
            self.model = AutoModel.from_pretrained(torch_dtype=dtype, **kwargs)

        self._adapter_keys = None
        if cfg.tune_mode == "lora" and getattr(cfg, "adapter_path", None):
            self._adapter_keys = _saved_adapter_keys(cfg.adapter_path)

        if getattr(self.model.config, "model_type", None) == "qwen3_5":
            layout = (_qwen35_adapter_layout(self._adapter_keys)
                      if self._adapter_keys is not None else "qwen35_text_only")
            if layout == "qwen35_text_only":
                self.model = self.model.language_model
            else:
                self.model.language_model.config.use_cache = False

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        if cfg.tune_mode == "frozen":
            for p in self.model.parameters():
                p.requires_grad_(False)
        elif cfg.tune_mode == "lora":
            self._attach_lora(cfg)
        elif cfg.tune_mode == "full":
            for p in self.model.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(f"unknown tune_mode={cfg.tune_mode}")

        if cfg.gradient_checkpointing and cfg.tune_mode != "frozen":
            if hasattr(self.model, "gradient_checkpointing_enable"):


                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        self.hidden_size = self._hidden_size()

    def _attach_lora(self, cfg):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            raise RuntimeError("LoRA requested but peft is not installed") from e

        if cfg.adapter_path:
            self.model = _load_lora_strict(self.model, cfg.adapter_path, self._adapter_keys)
            return

        lcfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.model = get_peft_model(self.model, lcfg)

    def _hidden_size(self) -> int:
        c = self.model.config
        for name in ("hidden_size", "d_model", "n_embd"):
            if hasattr(c, name):
                return int(getattr(c, name))
        if hasattr(c, "text_config") and hasattr(c.text_config, "hidden_size"):
            return int(c.text_config.hidden_size)
        raise ValueError("could not infer backbone hidden size")

    def train(self, mode: bool = True):
        super().train(mode)


        if self.cfg.tune_mode == "frozen":
            self.model.eval()
        return self

    def _forward_hidden(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(**inputs, return_dict=True)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None and hasattr(out, "hidden_states") and out.hidden_states:
            hidden = out.hidden_states[-1]
        if hidden is None:
            raise RuntimeError("backbone did not return last_hidden_state")
        return hidden

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.cfg.tune_mode == "frozen":
            with torch.no_grad():
                hidden = self._forward_hidden(inputs)
        else:
            hidden = self._forward_hidden(inputs)

        mask = inputs["attention_mask"]
        pooling = self.cfg.pooling

        if pooling == "last":
            last = mask.long().sum(dim=-1).sub(1).clamp_min(0)
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[batch, last]

        if pooling == "cls":
            return hidden[:, 0]

        if pooling == "mean":
            w = mask.to(dtype=hidden.dtype).unsqueeze(-1)
            denom = w.sum(dim=1).clamp_min(1.0)
            return (hidden * w).sum(dim=1) / denom

        raise ValueError(f"unknown pooling={pooling}")

    def save_adapter(self, path: str | Path):
        if self.cfg.tune_mode != "lora":
            return
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)

    def save_full_backbone(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
