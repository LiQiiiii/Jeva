from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from time import perf_counter

import torch

from .types import Answer, Choice, DecisionResponse, Noul, RankResult, RankedItem, Score, Usage
from ._runtime.cache import BoundedCache
from ._runtime.collate import option_text, state_question_text, tokenize_texts
from ._runtime.config import Config, load_config


class Jeva:


    def __init__(self, model, config: Config, *, device="cuda", cache="hybrid",
                 max_state_length=None, max_option_length=None, truncation="error",
                 option_chunk_size=16, option_batch_tokens=16384,
                 option_cache_size=4096, option_cache_max_bytes=256 << 20,
                 set_cache_size=64, set_cache_max_bytes=256 << 20):
        if cache not in {"none", "set", "hybrid"}:
            raise ValueError("cache must be 'none', 'set', or 'hybrid'")
        if truncation not in {"error", "truncate"}:
            raise ValueError("truncation must be 'error' or 'truncate'")
        if any(type(v) is not int or v < 1 for v in (option_chunk_size, option_batch_tokens)):
            raise ValueError("option_chunk_size and option_batch_tokens must be positive")
        self.model, self.config = model, config
        self.device = torch.device(device)
        self.cache = cache
        self.max_state_length = config.data.max_state_length if max_state_length is None else int(max_state_length)
        self.max_option_length = config.data.max_option_length if max_option_length is None else int(max_option_length)
        if min(self.max_state_length, self.max_option_length) < 1:
            raise ValueError("maximum token lengths must be positive")
        self.truncation = truncation
        self.option_chunk_size, self.option_batch_tokens = option_chunk_size, option_batch_tokens
        self._options = BoundedCache(option_cache_size, option_cache_max_bytes)
        self._sets = BoundedCache(set_cache_size, set_cache_max_bytes)
        self._lock = RLock()
        self.checkpoint_dir: Path | None = None
        self.model.to(self.device).eval().requires_grad_(False)

    @classmethod
    def from_pretrained(cls, checkpoint="LIQIIIII/Jeva-4B", *, device="cuda",
                        merge_lora=True, cache="hybrid", dtype=None, revision=None,
                        local_files_only=False, trust_remote_code=False, **kwargs):

        from ._runtime.model import System1DecisionModel

        if not isinstance(merge_lora, bool):
            raise ValueError("merge_lora must be a bool")
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select device='cpu' or use a CUDA environment")
        directory = Path(checkpoint).expanduser()
        if not directory.is_dir():
            if directory.is_absolute() or str(checkpoint).startswith("."):
                raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
            from huggingface_hub import snapshot_download
            from .download import INFERENCE_FILES
            directory = Path(snapshot_download(
                repo_id=str(checkpoint), revision=revision, local_files_only=local_files_only,
                allow_patterns=INFERENCE_FILES,
            ))
        directory = directory.resolve()
        config = load_config(directory / "config.yaml")
        cfg = config.model
        for field in ("backbone", "tokenizer_path", "head_path", "adapter_path"):
            value = getattr(cfg, field)
            if value and not Path(value).is_absolute() and (directory / value).exists():
                setattr(cfg, field, str((directory / value).resolve()))
        if (directory / "backbone_full").is_dir():
            cfg.backbone = str(directory / "backbone_full")
            cfg.tune_mode, cfg.adapter_path = "full", None
        elif (directory / "backbone_adapter").is_dir():
            cfg.adapter_path = str(directory / "backbone_adapter")
            cfg.tune_mode = "lora"
        if (directory / "tokenizer").is_dir():
            cfg.tokenizer_path = str(directory / "tokenizer")
        cfg.head_path = str(directory / "head.pt")
        if not Path(cfg.head_path).is_file():
            raise FileNotFoundError(f"Missing checkpoint head: {cfg.head_path}")
        if cfg.tune_mode == "lora" and not cfg.adapter_path:
            raise FileNotFoundError("LoRA checkpoint is missing backbone_adapter")
        cfg.gradient_checkpointing = False
        cfg.trust_remote_code = bool(trust_remote_code)
        cfg.local_files_only = bool(local_files_only)
        if dtype is not None:
            dtype_name = str(dtype).removeprefix("torch.")
            if dtype_name not in {"float32", "float16", "bfloat16"}:
                raise ValueError("dtype must be float32, float16, or bfloat16")
            cfg.torch_dtype = dtype_name
        model = System1DecisionModel(cfg).to(target_device).eval()
        if merge_lora and cfg.tune_mode == "lora":
            with torch.no_grad():
                model.backbone.model = model.backbone.model.merge_and_unload(safe_merge=True)
        instance = cls(model, config, device=target_device, cache=cache, **kwargs)
        instance.checkpoint_dir = directory
        instance.lora_merged = bool(merge_lora and cfg.tune_mode == "lora")
        return instance

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def clear_cache(self):
        with self._lock:
            self._options.clear()
            self._sets.clear()

    def cache_info(self):
        with self._lock:
            return {"mode": self.cache, "options": self._options.info(), "sets": self._sets.info()}

    @staticmethod
    def _state_text(state):
        if isinstance(state, str):
            return state
        return json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)

    def _batch(self, rows):
        batch = self.model.tokenizer.pad(rows, padding=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in batch.items()}

    def _encode_options(self, rows):
        outputs = []
        start = 0
        while start < len(rows):
            end, longest = start, 0
            while end < len(rows) and end - start < self.option_chunk_size:
                candidate = max(longest, len(rows[end]["input_ids"]))
                if end > start and candidate * (end + 1 - start) > self.option_batch_tokens:
                    break
                longest = candidate
                end += 1
            outputs.append(self.model.encode_option_inputs(self._batch(rows[start:end])))
            start = end
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _criteria(criteria):
        if isinstance(criteria, Mapping):
            keys = [str(key) for key in criteria]
            descriptions = list(criteria.values())
            if not all(isinstance(value, str) and value.strip() for value in descriptions):
                raise ValueError("criteria descriptions must be nonempty strings")
            texts = [f"{key}: {description}" for key, description in zip(keys, descriptions, strict=True)]
        elif isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
            keys = list(criteria)
            texts = keys.copy()
        else:
            raise TypeError("criteria must be a mapping or a sequence of strings")
        if not keys or not all(isinstance(key, str) and key.strip() for key in keys):
            raise ValueError("criteria must contain nonempty string labels")
        if len(set(keys)) != len(keys):
            raise ValueError("criteria labels must be unique")
        return keys, texts

    def _question(self, question):
        if isinstance(question, Noul):
            return question.instructions, "noul", ["false", "true"], ["false", "true"], False, None, 1
        if not isinstance(question, (Choice, Score)):
            raise TypeError("questions must be Choice, Noul, or Score objects")
        labels, options = self._criteria(question.criteria)
        if isinstance(question, Choice):
            return question.instructions, "choice", labels, options, question.allow_null, None, None
        values = question.values
        if values is None:
            if isinstance(question.criteria, Mapping):
                try:
                    values = [float(key) for key in labels]
                except ValueError:
                    values = list(range(len(labels)))
            else:
                values = list(range(len(labels)))
        return question.instructions, "score", labels, options, False, values, None

    def _predict(self, state, instructions, kind, labels, options, allow_null, values, positive_index):
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("instructions must be a nonempty string")
        if kind not in {"choice", "noul", "score"}:
            raise ValueError("decision_type must be choice, noul, or score")
        if not isinstance(allow_null, bool):
            raise ValueError("allow_null must be a bool")
        if not options or not all(isinstance(text, str) and text.strip() for text in options):
            raise ValueError("options must contain nonempty strings")
        if kind == "noul":
            if len(options) != 2 or type(positive_index) is not int or positive_index not in {0, 1}:
                raise ValueError("noul requires two options and positive_index 0 or 1")
        if kind == "score":
            if values is not None and (not isinstance(values, Sequence) or isinstance(values, (str, bytes))):
                raise ValueError("score values must be a sequence of finite numbers")
            values = list(range(len(options))) if values is None else list(values)
            if len(values) != len(options) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in values
            ):
                raise ValueError("score values must match criteria and be finite")
        if self.model.training:
            raise RuntimeError("Inference requires eval mode; clear caches after training before reuse")
        self._sync()
        started = perf_counter()
        state_rows, state_lengths, truncated_state = tokenize_texts(
            self.model.tokenizer, [state_question_text(state, instructions, kind)],
            self.max_state_length, self.truncation, field="state",
        )
        texts = [option_text(text, kind) for text in options]
        rows, lengths, truncated_options = tokenize_texts(
            self.model.tokenizer, texts, self.max_option_length, self.truncation, field="option",
        )


        keys = [tuple(row["input_ids"]) for row in rows]
        set_key = tuple(keys)
        encoded_options = 0
        with torch.inference_mode():
            c = self.model.encode_state_inputs(self._batch(state_rows))
            option_cache = self._sets.get(set_key) if self.cache != "none" else None
            if option_cache is None:
                if self.cache == "hybrid":
                    resolved, missing = {}, {}
                    for index, key in enumerate(keys):
                        if key in resolved or key in missing:
                            continue
                        memory = self._options.get(key)
                        if memory is None:
                            missing[key] = index
                        else:
                            resolved[key] = memory
                    if missing:
                        indices = list(missing.values())
                        u = self._encode_options([rows[i] for i in indices])
                        encoded_options = sum(lengths[i] for i in indices)
                        for key, memory in zip(missing, self.model.compile_option_memory(u), strict=True):
                            resolved[key] = memory
                            self._options.put(key, memory)
                    option_cache = self.model.assemble_option_memory([resolved[key] for key in keys])
                else:
                    u = self._encode_options(rows)
                    encoded_options = sum(lengths)
                    mask = torch.ones((1, len(options)), dtype=torch.bool, device=self.device)
                    option_cache = self.model.prepare_option_kv_cache(u.unsqueeze(0), mask)
                if self.cache != "none":
                    self._sets.put(set_key, option_cache)
            out = self.model.decide_from_option_cache(c, option_cache)
        q = out["q"][0].float().cpu().tolist()
        p_null = float(out["p_null"][0].float().cpu()) if allow_null else 0.0
        if (len(q) != len(options) or not all(math.isfinite(value) and 0 <= value <= 1 for value in [*q, p_null])
                or not math.isclose(sum(q), 1.0, abs_tol=1e-5)):
            raise RuntimeError("Model returned invalid probabilities")
        probabilities = [(1.0 - p_null) * value for value in q]
        best = max(range(len(q)), key=q.__getitem__)
        usage = Usage(
            input_tokens=state_lengths[0] + encoded_options,
            requested_input_tokens=state_lengths[0] + sum(lengths),
            encoded_state_tokens=state_lengths[0], encoded_option_tokens=encoded_options,
            cached_option_tokens=sum(lengths) - encoded_options,
            truncated_tokens=truncated_state + truncated_options,
        )
        self._sync()
        return Answer(
            kind=kind, probabilities=dict(zip(labels, probabilities, strict=True)),
            conditional_probabilities=dict(zip(labels, q, strict=True)), null_probability=p_null,
            choice=(None if allow_null and p_null >= probabilities[best] else labels[best]) if kind == "choice" else None,
            noul=q[positive_index] if kind == "noul" else None,
            score=sum(p * float(v) for p, v in zip(q, values, strict=True)) if kind == "score" else None,
            confidence=max(p_null, max(probabilities)), candidate_probabilities=tuple(q),
            usage=usage, latency_ms=(perf_counter() - started) * 1000,
        )

    def predict(self, state, question, decision_type, options, *, allow_null=False,
                values=None, positive_index=1):

        with self._lock:
            if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
                raise TypeError("options must be a sequence of strings")
            options = list(options)
            return self._predict(self._state_text(state), question, decision_type,
                                 [str(i) for i in range(len(options))], options,
                                 allow_null, values, positive_index)

    def system_one(self, state, questions):

        with self._lock:
            named = isinstance(questions, Mapping)
            items = list(questions.items()) if named else list(enumerate(questions))
            if not items:
                raise ValueError("questions must not be empty")
            if named and not all(isinstance(name, str) for name, _ in items):
                raise TypeError("question names must be strings")
            prepared = [(name, self._question(question)) for name, question in items]
            text = self._state_text(state)
            self._sync()
            started = perf_counter()
            results = [(name, self._predict(text, *spec)) for name, spec in prepared]
            usage = Usage()
            for _, answer in results:
                for key in usage.__dataclass_fields__:
                    setattr(usage, key, getattr(usage, key) + getattr(answer.usage, key))
            self._sync()
            return DecisionResponse(dict(results) if named else [answer for _, answer in results],
                                    usage, (perf_counter() - started) * 1000)

    def rank(self, state, candidates, *, instructions="Rank candidates by relevance to the state.",
             allow_null=False, top_k=None):

        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
            raise ValueError("top_k must be a positive integer")
        response = self.system_one(state, [Choice(instructions, candidates, allow_null)])
        answer = response.answers[0]
        ranking = [RankedItem(label, probability, i)
                   for i, (label, probability) in enumerate(answer.probabilities.items())]
        ranking.sort(key=lambda item: -item.probability)
        return RankResult(ranking[:top_k], answer.null_probability, response.usage, response.latency_ms)
