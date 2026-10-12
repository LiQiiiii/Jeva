from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: Mapping[str, str] | Sequence[str]
    allow_null: bool = False


@dataclass(frozen=True)
class Noul:

    instructions: str


@dataclass(frozen=True)
class Score:

    instructions: str
    criteria: Mapping[str | int | float, str] | Sequence[str]
    values: Sequence[float] | None = None


Question = Choice | Noul | Score


@dataclass
class Usage:

    input_tokens: int = 0
    requested_input_tokens: int = 0
    encoded_state_tokens: int = 0
    encoded_option_tokens: int = 0
    cached_option_tokens: int = 0
    output_tokens: int = 0
    truncated_tokens: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class Answer:
    kind: str
    probabilities: dict[str, float]
    conditional_probabilities: dict[str, float]
    null_probability: float
    choice: str | None = None
    noul: float | None = None
    score: float | None = None
    confidence: float = 0.0
    candidate_probabilities: tuple[float, ...] = ()
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class DecisionResponse:
    answers: dict[str, Answer] | list[Answer]
    usage: Usage
    latency_ms: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RankedItem:
    label: str
    probability: float
    index: int


@dataclass
class RankResult:
    ranking: list[RankedItem]
    null_probability: float
    usage: Usage
    latency_ms: float

    def to_dict(self):
        return asdict(self)
