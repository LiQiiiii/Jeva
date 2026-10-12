from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainingRecord:
    state: str
    question: str
    decision_type: str
    options: tuple[str, ...]
    target_probs: tuple[float, ...]
    allow_null: bool = False
    values: tuple[float, ...] | None = None
    positive_index: int | None = None


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def normalize_record(row: Mapping[str, Any]) -> TrainingRecord:

    kind = row.get("decision_type", "choice")
    if kind not in {"choice", "noul", "score"}:
        raise ValueError("decision_type must be choice, noul, or score")
    state, question = row.get("state"), row.get("question")
    if not isinstance(state, str) or not isinstance(question, str) or not question.strip():
        raise ValueError("state and nonempty question must be strings")
    options = row.get("options", ["false", "true"] if kind == "noul" else None)
    if not isinstance(options, (list, tuple)) or not options:
        raise ValueError("options must be a nonempty list of strings")
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("every option must be a nonempty string")
    k = len(options)
    allow_null = row.get("allow_null", False)
    if not isinstance(allow_null, bool):
        raise ValueError("allow_null must be a boolean")
    if kind != "choice" and allow_null:
        raise ValueError("allow_null is supported only for choice records")
    supplied = [name for name in ("target_index", "target_probs", "p_true", "score") if name in row]
    if len(supplied) != 1:
        raise ValueError("provide exactly one of target_index, target_probs, p_true, or score")
    label = supplied[0]
    values, positive_index = None, None
    if kind == "noul":
        if k != 2:
            raise ValueError("noul requires exactly two options")
        positive_index = row.get("positive_index", 1)
        if type(positive_index) is not int or positive_index not in (0, 1):
            raise ValueError("positive_index must be 0 or 1")
    if kind == "score":
        raw_values = row.get("values", list(range(k)))
        if not isinstance(raw_values, (list, tuple)) or len(raw_values) != k or k < 2:
            raise ValueError("score requires at least two options and one value per option")
        values = tuple(_finite(value, "values") for value in raw_values)
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError("score values must be strictly increasing in option order")
    probs = [0.0] * (k + 1)
    if label == "target_probs":
        raw_probs = row[label]
        if not isinstance(raw_probs, (list, tuple)) or len(raw_probs) != k + 1:
            raise ValueError("target_probs must contain null probability followed by one per option")
        probs = [_finite(value, "target_probs") for value in raw_probs]
        if min(probs) < 0 or not math.isclose(sum(probs), 1.0, rel_tol=0, abs_tol=1e-5):
            raise ValueError("target_probs must be nonnegative and sum to one")
        total = sum(probs)
        probs = [value / total for value in probs]
    elif label == "target_index":
        index = row[label]
        if index is None:
            if not allow_null:
                raise ValueError("a null target requires allow_null=true")
            probs[0] = 1.0
        elif type(index) is not int or not 0 <= index < k:
            raise ValueError("target_index must be a zero-based option index or null")
        else:
            probs[index + 1] = 1.0
    elif label == "p_true":
        if kind != "noul":
            raise ValueError("p_true is only valid for noul")
        probability = _finite(row[label], label)
        if not 0 <= probability <= 1:
            raise ValueError("p_true must be between zero and one")
        probs[positive_index + 1] = probability
        probs[2 - positive_index] = 1.0 - probability
    else:
        if kind != "score":
            raise ValueError("score is only valid for score records")
        score = _finite(row[label], label)
        if not values[0] <= score <= values[-1]:
            raise ValueError("score must fall within the supplied rubric values")
        for index, value in enumerate(values):
            if score <= value:
                if index == 0:
                    probs[1] = 1.0
                else:
                    fraction = (score - values[index - 1]) / (value - values[index - 1])
                    probs[index] = 1.0 - fraction
                    probs[index + 1] = fraction
                break
    if not allow_null and probs[0] != 0:
        raise ValueError("null target mass requires allow_null=true")
    return TrainingRecord(state, question, kind, tuple(options), tuple(probs),
                          allow_null, values, positive_index)


def read_jsonl(path: str | Path) -> list[TrainingRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each line must be a JSON object")
                records.append(normalize_record(row))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid training record at line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("the dataset is empty")
    return records


def read_hf_dataset(dataset: str, *, split: str, columns: Mapping[str, str],
                    name: str | None = None) -> list[TrainingRecord]:

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face datasets support requires pip install 'jeva-model[train]'") from exc
    if not split:
        raise ValueError("an explicit dataset split is required")
    if not columns:
        raise ValueError("supply a column map from Jeva field names to dataset columns")
    dataset_rows = load_dataset(dataset, name=name, split=split)
    records = [normalize_record({field: row[column] for field, column in columns.items()})
               for row in dataset_rows]
    if not records:
        raise ValueError("the dataset split is empty")
    return records


def _digest(records: Iterable[TrainingRecord], *, labels: bool = True) -> str:
    digest = hashlib.sha256()
    for record in records:
        item = asdict(record)
        if not labels:
            for key in ("target_probs", "positive_index", "allow_null", "values"):
                item.pop(key)
            item["options"] = sorted(item["options"])
        digest.update(json.dumps(item, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8") + b"\n")
    return digest.hexdigest()


def validate_holdout(train: list[TrainingRecord], validation: list[TrainingRecord]) -> None:
    seen = {_digest([row], labels=False) for row in train}
    if any(_digest([row], labels=False) in seen for row in validation):
        raise ValueError("validation data overlaps training tasks; supply a separate holdout")


class TrainingCollator:
    def __init__(self, tokenizer, max_state_length: int, max_option_length: int,
                 truncation: str = "error"):
        self.tokenizer = tokenizer
        self.max_state_length = max_state_length
        self.max_option_length = max_option_length
        self.truncation = truncation
        self.truncated_state_tokens = 0
        self.truncated_option_tokens = 0

    def __call__(self, rows: list[TrainingRecord]) -> dict[str, Any]:
        from ._runtime.collate import option_text, state_question_text, tokenize_texts
        counts = [len(row.options) for row in rows]
        kmax = max(counts)
        state_texts = [state_question_text(row.state, row.question, row.decision_type) for row in rows]
        option_texts = [option_text(option, row.decision_type) for row in rows for option in row.options]
        state_rows, _, state_truncated = tokenize_texts(
            self.tokenizer, state_texts, self.max_state_length, self.truncation, field="state")
        option_rows, _, option_truncated = tokenize_texts(
            self.tokenizer, option_texts, self.max_option_length, self.truncation, field="option")
        state_inputs = self.tokenizer.pad(state_rows, padding=True, return_tensors="pt")
        option_inputs = self.tokenizer.pad(option_rows, padding=True, return_tensors="pt")
        self.truncated_state_tokens += state_truncated
        self.truncated_option_tokens += option_truncated
        mask = torch.zeros((len(rows), kmax), dtype=torch.bool)
        target = torch.zeros((len(rows), kmax + 1), dtype=torch.float32)
        for index, row in enumerate(rows):
            mask[index, :counts[index]] = True
            target[index, :counts[index] + 1] = torch.tensor(row.target_probs)
        return {"state_inputs": dict(state_inputs), "option_inputs": dict(option_inputs),
                "option_counts": torch.tensor(counts), "option_mask": mask,
                "target_probs": target,
                "allow_null": torch.tensor([row.allow_null for row in rows], dtype=torch.bool)}


def supervised_loss(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> torch.Tensor:

    mask = batch["option_mask"]
    candidates = batch["target_probs"][:, 1:].float()

    log_q = outputs["log_q"].float().masked_fill(~mask, 0.0)
    candidate_loss = -(candidates * log_q).sum(dim=-1)
    coverage = candidates.sum(dim=-1)
    allow_null = batch["allow_null"]
    coverage_loss = candidate_loss.new_zeros(candidate_loss.shape)
    if bool(allow_null.any()):
        coverage_loss[allow_null] = F.binary_cross_entropy_with_logits(
            outputs["coverage_logit"][allow_null].float(), coverage[allow_null], reduction="none")
    return (candidate_loss + coverage_loss).mean()


@dataclass(frozen=True)
class TrainOptions:
    mode: str = "head"
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    max_state_length: int | None = None
    max_option_length: int | None = None
    truncation: str = "error"
    option_chunk_size: int = 8
    seed: int = 42
    device: str = "cpu"

    def validate(self) -> None:
        if self.mode not in {"head", "lora"}:
            raise ValueError("mode must be head or lora")
        if self.truncation not in {"error", "truncate"}:
            raise ValueError("truncation must be error or truncate")
        for name in ("max_state_length", "max_option_length"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        for name in ("epochs", "batch_size", "gradient_accumulation", "option_chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")


def configure_trainable(model: nn.Module, mode: str) -> list[tuple[str, nn.Parameter]]:
    if mode not in {"head", "lora"}:
        raise ValueError("mode must be head or lora")
    model.requires_grad_(False)
    for module in (model.context_proj, model.option_proj, model.core):
        module.requires_grad_(True)
    model.backbone.cfg.tune_mode = "frozen" if mode == "head" else "lora"
    if mode == "lora":
        lora_parameters = [(name, param) for name, param in model.backbone.model.named_parameters()
                           if ".lora_" in name]
        if not lora_parameters:
            raise ValueError("lora mode requires a Jeva checkpoint containing a loaded LoRA adapter")
        for _, parameter in lora_parameters:
            parameter.requires_grad_(True)
        if hasattr(model.backbone.model, "gradient_checkpointing_enable"):
            model.backbone.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model.backbone.model, "enable_input_require_grads"):
            model.backbone.model.enable_input_require_grads()
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: _to_device(value, device) if isinstance(value, dict)
            else value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def train_epoch(model: nn.Module, records: list[TrainingRecord], collator: TrainingCollator,
                optimizer: torch.optim.Optimizer, options: TrainOptions, epoch: int) -> float:
    model.train()
    if options.mode == "head":
        model.backbone.eval()
    ordered = list(range(len(records)))
    random.Random(options.seed + epoch).shuffle(ordered)
    batches = [ordered[start:start + options.batch_size]
               for start in range(0, len(ordered), options.batch_size)]
    total = 0.0
    device = torch.device(options.device)
    for group_start in range(0, len(batches), options.gradient_accumulation):
        group = batches[group_start:group_start + options.gradient_accumulation]
        group_size = sum(map(len, group))
        optimizer.zero_grad(set_to_none=True)
        for indices in group:
            batch = _to_device(collator([records[index] for index in indices]), device)
            outputs = model(batch, option_chunk_size=options.option_chunk_size)
            loss = supervised_loss(outputs, batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("training produced a nonfinite loss; no checkpoint was written")
            (loss * (len(indices) / group_size)).backward()
            total += float(loss.detach()) * len(indices)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                      options.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
    return total / len(records)


@torch.no_grad()
def evaluate_loss(model: nn.Module, records: list[TrainingRecord], collator: TrainingCollator,
                  options: TrainOptions) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(records), options.batch_size):
        rows = records[start:start + options.batch_size]
        batch = _to_device(collator(rows), torch.device(options.device))
        outputs = model(batch, option_chunk_size=options.option_chunk_size)
        total += float(supervised_loss(outputs, batch)) * len(rows)
    return total / len(records)


def _resume_settings(options: TrainOptions) -> dict[str, Any]:
    result = asdict(options)
    result.pop("epochs")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _remove_source_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _remove_source_paths(item) for key, item in value.items()
                if key not in {"_name_or_path", "name_or_path"}}
    if isinstance(value, list):
        return [_remove_source_paths(item) for item in value]
    return value


def _snapshot_backbone(source: str, destination: Path, revision: str | None = None) -> str:
    source_path = Path(source)
    if not source_path.is_dir():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source):
            raise ValueError("backbone must resolve to a local safetensors directory or a Hub repository ID")
        from huggingface_hub import snapshot_download
        source_path = Path(snapshot_download(
            repo_id=source, revision=revision,
            allow_patterns=["config.json", "generation_config.json", "*.safetensors", "*.safetensors.index.json"]))
    if source_path.is_dir():


        assets = [path for path in source_path.iterdir() if path.is_file() and
                  (path.name in {"config.json", "generation_config.json"} or
                   path.name.endswith((".safetensors", ".safetensors.index.json")))]
        if not any(path.name.endswith(".safetensors") for path in assets):
            raise ValueError("local backbone export requires safetensors weights")
        destination.mkdir()
        for path in assets:
            if path.suffix == ".json":
                content = json.loads(path.read_text(encoding="utf-8"))

                _write_json(destination / path.name, _remove_source_paths(content))
            else:
                shutil.copy2(path, destination / path.name)
        return "backbone"
    raise ValueError("backbone snapshot could not be resolved")


def save_training_checkpoint(wrapper, output: str | Path, optimizer, options: TrainOptions,
                             train: list[TrainingRecord], validation: list[TrainingRecord],
                             history: list[dict[str, Any]], names: list[str]) -> Path:

    import yaml
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    model = wrapper.model
    config = copy.deepcopy(wrapper.config.to_dict())
    cfg = config["model"]
    revision = getattr(getattr(model.backbone.model, "config", None), "_commit_hash", None)
    cfg["backbone"] = _snapshot_backbone(model.backbone.cfg.backbone, out / "backbone", revision)
    cfg["head_path"] = "head.pt"
    cfg["tokenizer_path"] = "tokenizer"
    cfg["gradient_checkpointing"] = options.mode == "lora"
    config["data"] = {"max_state_length": options.max_state_length,
                      "max_option_length": options.max_option_length}
    model.save_head(out / "head.pt")
    model.tokenizer.save_pretrained(out / "tokenizer")
    tokenizer_config = out / "tokenizer" / "tokenizer_config.json"
    if tokenizer_config.exists():
        content = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        _write_json(tokenizer_config, _remove_source_paths(content))
    encoder = model.backbone.model
    if hasattr(encoder, "peft_config"):
        encoder.save_pretrained(out / "backbone_adapter", safe_serialization=True)
        (out / "backbone_adapter" / "README.md").write_text(
            "# Jeva adapter\n\nLoad this adapter through the parent Jeva checkpoint.\n", encoding="utf-8")
        adapter_config = out / "backbone_adapter" / "adapter_config.json"
        content = json.loads(adapter_config.read_text(encoding="utf-8"))
        content["base_model_name_or_path"] = cfg["backbone"]
        _write_json(adapter_config, content)

        cfg["tune_mode"] = "lora"
        cfg["adapter_path"] = "backbone_adapter"
    else:
        cfg["tune_mode"] = "frozen"
        cfg["adapter_path"] = None
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    state = {"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(state, out / "trainer_state.pt")
    manifest = {"format_version": 1, "completed_epochs": options.epochs,
                "settings": _resume_settings(options), "train_digest": _digest(train),
                "validation_digest": _digest(validation), "parameter_names": names,
                "history": history,
                "files": {str(path.relative_to(out)): _file_sha256(path)
                          for path in out.rglob("*") if path.is_file()}}
    _write_json(out / "training.json", manifest)
    return out


def _load_resume(path: Path, options: TrainOptions, train: list[TrainingRecord],
                 validation: list[TrainingRecord], names: list[str], optimizer) -> dict[str, Any]:
    manifest = json.loads((path / "training.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported training checkpoint format")
    for key, expected in (("settings", _resume_settings(options)), ("train_digest", _digest(train)),
                          ("validation_digest", _digest(validation)), ("parameter_names", names)):
        if manifest.get(key) != expected:
            raise ValueError(f"resume verification failed: {key} differs")
    completed = manifest.get("completed_epochs")
    if type(completed) is not int or not 0 <= completed < options.epochs:
        raise ValueError("epochs must exceed the checkpoint's completed epoch count")
    files = manifest.get("files", {})
    if not {"config.yaml", "head.pt", "trainer_state.pt"}.issubset(files):
        raise ValueError("resume checkpoint has an incomplete file manifest")
    for relative, expected_hash in files.items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file():
            raise ValueError("resume manifest contains a missing or external file")
        if _file_sha256(target) != expected_hash:
            raise ValueError(f"resume checkpoint checksum mismatch: {relative}")
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if not torch.cuda.is_available() or len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("resume requires the saved CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return manifest


def fine_tune(checkpoint: str | Path, train_data: str | Path | Iterable[Mapping[str, Any] | TrainingRecord],
              output_dir: str | Path, *, validation_data=None, options: TrainOptions | None = None,
              resume: bool = False) -> Path:

    from . import Jeva
    options = options or TrainOptions()
    options.validate()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("output_dir already exists; choose a new directory")
    source = Path(checkpoint)
    if resume and (not source.is_dir() or not (source / "training.json").is_file()):
        raise ValueError("resume requires a local completed training output containing training.json")
    if source.exists() and (output_dir.resolve() == source.resolve()
                            or source.resolve() in output_dir.resolve().parents):
        raise ValueError("output_dir must be outside the source checkpoint")

    def records(data):
        if isinstance(data, (str, Path)):
            return read_jsonl(data)
        return [normalize_record(asdict(row) if isinstance(row, TrainingRecord) else row) for row in data]

    train = records(train_data)
    validation = records(validation_data) if validation_data is not None else []
    if not train:
        raise ValueError("training data is empty")
    if validation_data is not None and not validation:
        raise ValueError("validation data is empty")
    validate_holdout(train, validation)
    torch.manual_seed(options.seed)
    wrapper = Jeva.from_pretrained(str(checkpoint), device=options.device,
                                  merge_lora=False, cache="none")
    options = replace(options,
                      max_state_length=options.max_state_length or wrapper.config.data.max_state_length,
                      max_option_length=options.max_option_length or wrapper.config.data.max_option_length)

    clear_cache = getattr(wrapper, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()
    model = wrapper.model
    parameters = configure_trainable(model, options.mode)
    names = [name for name, _ in parameters]
    optimizer = torch.optim.AdamW([parameter for _, parameter in parameters],
                                 lr=options.learning_rate, weight_decay=options.weight_decay)
    collator = TrainingCollator(model.tokenizer, options.max_state_length,
                                options.max_option_length, options.truncation)
    start_epoch, history = 0, []
    if resume:
        manifest = _load_resume(wrapper.checkpoint_dir, options, train, validation, names, optimizer)
        start_epoch, history = manifest["completed_epochs"], manifest["history"]
    for epoch in range(start_epoch, options.epochs):
        state_before, option_before = collator.truncated_state_tokens, collator.truncated_option_tokens
        loss = train_epoch(model, train, collator, optimizer, options, epoch)
        metrics = {"epoch": epoch + 1, "train_loss": loss}
        if options.truncation == "truncate":
            metrics["train_truncated_state_tokens"] = collator.truncated_state_tokens - state_before
            metrics["train_truncated_option_tokens"] = collator.truncated_option_tokens - option_before
        if validation:
            state_before, option_before = collator.truncated_state_tokens, collator.truncated_option_tokens
            metrics["validation_loss"] = evaluate_loss(model, validation, collator, options)
            if options.truncation == "truncate":
                metrics["validation_truncated_state_tokens"] = collator.truncated_state_tokens - state_before
                metrics["validation_truncated_option_tokens"] = collator.truncated_option_tokens - option_before
        history.append(metrics)
        print(json.dumps(metrics, allow_nan=False), flush=True)
    model.eval()
    if clear_cache is not None:
        clear_cache()
    return save_training_checkpoint(wrapper, output_dir, optimizer, options, train,
                                    validation, history, names)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Jeva on your own labeled examples")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Released Jeva checkpoint directory or Hub repository")
    source.add_argument("--resume-from", help="Completed training output to resume into a new directory")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--train-data", help="Your training JSONL")
    data.add_argument("--dataset", help="Hugging Face dataset repository or local dataset")
    parser.add_argument("--dataset-name")
    parser.add_argument("--train-split", help="Explicit HF training split")
    parser.add_argument("--validation-split", help="Explicit separate HF validation split")
    parser.add_argument("--column-map", help='JSON object mapping canonical fields to HF columns')
    parser.add_argument("--validation-data", help="Separate validation JSONL; never selected automatically")
    parser.add_argument("--output-dir", required=True, help="New checkpoint directory")
    defaults = TrainOptions()
    for name in asdict(defaults):
        default = getattr(defaults, name)
        kwargs = {"default": default, "type": type(default) if default is not None else int}
        if name == "mode":
            kwargs["choices"] = ["head", "lora"]
        if name == "truncation":
            kwargs["choices"] = ["error", "truncate"]
        parser.add_argument("--" + name.replace("_", "-"), **kwargs)
    args = parser.parse_args(argv)
    if args.dataset:
        if not args.train_split or not args.column_map:
            parser.error("--dataset requires --train-split and --column-map")
        if args.validation_data and args.validation_split:
            parser.error("choose --validation-data or --validation-split")
        try:
            columns = json.loads(args.column_map)
            if not isinstance(columns, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                       for k, v in columns.items()):
                raise ValueError("--column-map must be an object of string values")
        except ValueError as exc:
            parser.error(str(exc))
        train = read_hf_dataset(args.dataset, name=args.dataset_name, split=args.train_split, columns=columns)
        validation = (read_hf_dataset(args.dataset, name=args.dataset_name, split=args.validation_split,
                                     columns=columns) if args.validation_split else args.validation_data)
    else:
        if args.train_split or args.validation_split or args.column_map or args.dataset_name:
            parser.error("HF mapping options require --dataset")
        train, validation = args.train_data, args.validation_data
    options = TrainOptions(**{name: getattr(args, name) for name in asdict(defaults)})
    output = fine_tune(args.resume_from or args.model, train, args.output_dir,
                       validation_data=validation, options=options, resume=bool(args.resume_from))
    print(f"Saved checkpoint: {output}")


if __name__ == "__main__":
    main()
