# Local evaluation

`jeva-evaluate` runs one typed decision per JSONL record. Prepared benchmark data
will be released separately on Hugging Face; the dataset link and download
instructions will be added when available. No data conversion, publication tools,
or saved experimental results are included in this repository.

## Run

```bash
jeva-evaluate \
  --model ./models/Jeva-4B \
  --data ./my_eval.jsonl \
  --output ./results/my_eval
```

JSONL and gzip-compressed `.jsonl.gz` are supported. If the records reference a
shared text corpus, supply `--corpus ./corpus.jsonl.gz` as well. Corpus entries
contain `id` and `text`; records may use `state_id`, `question_id`, and `option_ids`
instead of inline text. Candidate order is preserved.

For a Hub model, provide `--revision` with its immutable 40-character commit hash.
Local model files are content-hashed. Repeating a command resumes completed
records only when the model, inputs, settings, and implementation match. Use a
new output directory after changing any of them.

## Record format

```json
{"id":"example-1","state":"Please correct my invoice.","question":"Which team should handle this?","decision_type":"choice","options":["Billing","Technical support"],"allow_null":true,"target_probs":[0.0,1.0,0.0]}
```

Required fields are `id`, `state`, `question`, `decision_type`, `options`,
`allow_null`, and `target_probs`. At least two options are required.
`target_probs` contains null mass first, followed by one value per candidate;
values must be nonnegative and sum to one. Closed-set records have zero null mass.

- **Choice:** use `allow_null=true` when no listed candidate may apply.
- **Noul:** supply two options and `positive_index` (0 or 1), with `allow_null=false`.
- **Score:** supply numerical `values`, one per rubric option, with `allow_null=false`.

Evaluation labels are separate from model-visible inputs. Unlike the convenient
training format, evaluation records require an explicit target distribution.

For retrieval, add unique `candidate_ids`, complete `qrels`, and an optional
`eligible_grade`. Keep relevant documents outside the candidate pool in the qrels.

## Metrics

The runner saves `predictions.json`, `summary.json`, and the resume protocol in
the requested output directory. It rejects malformed or unsupported requests
rather than silently dropping them.

Accuracy defaults to `first_argmax`, including null where applicable. Set a row's
`accuracy_rule` to `tied_top_set` for multi-positive targets that accept any
highest-target-probability option. Soft targets are retained for NLL and Brier.
Noul reports errors against the positive-candidate target; Score reports MAE and
RMSE in the rubric's numerical units. Null detection and calibration are computed
where the supplied records support them.

Retrieval nDCG uses linear original relevance grades; eligible hits, MRR, and
recall use `eligible_grade`. Ranking ties preserve candidate order.

Ordinary evaluation starts each request with an empty option cache. Timings are
local request measurements, not a reproduction of every paper serving protocol.
The default limits are 4,096 state tokens and 512 option tokens, with errors on
overflow. Set `--max-state-length` and `--max-option-length` explicitly for longer
texts and use the same settings for comparisons.

## Candidate-order stability

```bash
jeva-evaluate \
  --model ./models/Jeva-4B \
  --data ./my_eval.jsonl \
  --permutations 3 --seed 20260929 \
  --output ./results/permutations
```

The original and reversed orders are included, plus up to three distinct random
orders. Candidate identities and Noul's positive index move with their options.
Score records receive ordinary evaluation but are excluded from permutation
metrics because their rubric is ordinal. The summary reports any-flip rate,
all-order accuracy, and candidate-aligned total variation. Exact score ties retain
display-order tie-breaking.

## Candidate-cache comparisons

The Python function `jeva.evaluation.evaluate_cache_episodes` compares `none`,
`set`, and `hybrid` caching on the same explicit episodes:

```python
from pathlib import Path
from jeva import Jeva
from jeva.evaluation import evaluate_cache_episodes, file_sha256, read_jsonl

checkpoint = Path("./models/Jeva-4B")
settings = {"device": "cuda", "max_state_length": 4096, "max_option_length": 512}
model = Jeva.from_pretrained(checkpoint, **settings)
requests = read_jsonl("./my_eval.jsonl")
report = evaluate_cache_episodes(
    model,
    episodes=[{"id": "stream-1", "requests": requests}],
    output="./results/cache",
    model_identity={
        "files": {
            str(p.relative_to(checkpoint)): file_sha256(p)
            for p in sorted(checkpoint.rglob("*"))
            if p.is_file() and ".cache" not in p.parts
        },
        "settings": settings,
    },
)
```

Keep all model components inside the local checkpoint directory. Record any
additional runtime overrides in `model_identity` when changing this example.
Each episode begins cold in every mode. An optional `prime` contains model-visible
fields but no targets; its time and cache counters are recorded separately.
Request timing excludes priming. Incomplete episodes restart cold on resume.
No previous answer is cached. Use new states to study candidate reuse rather than
repeating an identical full request.
