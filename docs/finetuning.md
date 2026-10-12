# Fine-tuning on your data

Jeva can continue learning from your labeled examples. Start with the released
checkpoint and choose `head` to train its decision projections and decision stack
while freezing the text encoder and its existing adapter. Choose `lora` to also
update the existing LoRA adapter. Neither mode trains the base encoder weights.

Install the training dependencies from the repository:

```bash
pip install -e '.[train]'
```

Before `--mode lora`, run `jeva-doctor --backward --strict`. The recommended
conda environment already includes the CUDA compiler required by the Hopper
TileLang backward path. If you created a Python-only environment instead, add
the matching compiler first:

```bash
conda install -n jeva -c nvidia/label/cuda-12.8.1 cuda-nvcc=12.8.93 -y
conda activate jeva
nvcc --version
jeva-doctor --backward --strict
```

Prepare your own training JSONL, then run:

```bash
python -m jeva.finetune \
  --model ./models/Jeva-4B \
  --train-data ./my_train.jsonl \
  --validation-data ./my_validation.jsonl \
  --output-dir ./outputs/my-jeva \
  --mode head --device cuda \
  --epochs 3 --batch-size 1 --gradient-accumulation 8 \
  --learning-rate 0.0001
```

`train/finetune.py` and `jeva-finetune` provide the same command. The four records
in `examples/finetune_data.jsonl` demonstrate the format; they are illustrative,
not a useful training corpus or validation set. Learning rate, length limits,
batch size, and epoch count are caller choices; the defaults are starting points
for experimentation, not performance guarantees.

The output directory must not exist and must sit outside the source checkpoint.
A checkpoint is saved after the requested epochs complete. There is no automatic
mid-epoch recovery. Keep enough free space for another copy of the frozen local
backbone, its adapter, tokenizer, head, and optimizer state.

## JSONL contract

Each line is a JSON object with `state` (text being assessed), `question`
(instructions), and `decision_type`: `choice`, `noul`, or `score`. Text templates
are identical to inference. Option descriptions should match the criteria you
intend to use at inference. When inference criteria are a mapping, render each
training option as `label: criterion` in the same insertion order.

Every record supplies exactly one label field: `target_index`, `target_probs`,
`p_true`, or `score`.

For a closed choice, supply a nonempty list of `options` and a zero-based index:

```json
{"decision_type":"choice","state":"Please correct the amount on my invoice.","question":"Select the support queue.","options":["billing: invoices and payments","access: sign-in and recovery"],"target_index":0}
```

For a choice that may reject every candidate, add `allow_null: true`. A JSON
`null` label means no listed option is acceptable:

```json
{"decision_type":"choice","state":"The requested service is unavailable.","question":"Which available service meets the request?","options":["Shipping","Repairs"],"allow_null":true,"target_index":null}
```

`target_probs` always lists the null probability first and then one probability
per option. Probabilities must be finite, nonnegative, and sum to one. Null mass
must be zero unless `allow_null` is true. A soft choice target might be
`[0.1, 0.6, 0.3]` for two candidates with null enabled.

For a Noul probability judgment, `p_true` accepts any value from zero to one:

```json
{"decision_type":"noul","state":"The response resolves most of the request.","question":"Does this response satisfy the request?","p_true":0.8}
```

Noul uses options `["false", "true"]` and `positive_index: 1` by default,
matching the inference API. Custom two-option formulations may supply `options`
and `positive_index` explicitly, and must use the same formulation at inference.
Noul has no null outcome.

For rubric scoring, options and numeric values must appear in increasing value
order. Values default to zero-based option indices if omitted:

```json
{"decision_type":"score","state":"The explanation is sound but misses an important example.","question":"Rate the explanation.","options":["Incorrect","Partly correct","Mostly correct","Complete"],"values":[0,1,2,3],"score":2.5}
```

A scalar score must fall within the rubric range. An intermediate value is
converted to a distribution over its two neighboring levels; `2.5` in the
example becomes `[0, 0, 0, 0.5, 0.5]`, including the leading null slot. Its
expected rubric value is exactly `2.5`. To supply a richer target distribution,
use `target_probs` instead. Scoring has no null outcome.

## Objective and batches

Training minimizes mean cross entropy. Closed choice, Noul, and score records
use the conditional candidate distribution. A choice with null enabled uses
the joint distribution: binary coverage cross entropy plus candidate cross
entropy weighted by the target's non-null mass. This allows both hard and soft
null labels without treating padding as a candidate. Score training fits the
rubric distribution; the inference API returns its expected numeric value.

Mixed option counts are masked. Gradient accumulation weights each example
equally, including a short final batch. Head training keeps the frozen encoder
in evaluation mode. LoRA training backpropagates through the encoder into its
existing adapter and the decision head. Inference caches are disabled during
training and are never used as training features.

`--max-state-length` and `--max-option-length` default to the checkpoint's saved
token limits; `--option-chunk-size` defaults to 8. Training rejects texts that
exceed those limits. To deliberately discard overflowing tokens, pass
`--truncation truncate`; each epoch then reports the number of removed state and
option tokens separately for training and validation. Inspect your text lengths
and increase the limits when truncation would discard essential information.
Smaller chunks reduce the size of individual option-encoder
forwards; they do not discard options. Length limits are saved into the output
checkpoint. This trainer uses a single device; it does not implement distributed
training, automatic batch sizing, or model selection.

## Validation and dataset mapping

Validation is optional and must be explicitly supplied as a separate file or
dataset split. The trainer never discovers benchmark data, reuses evaluation
folders, or creates an implicit train/validation split. It rejects exact task
overlap across supplied train and validation records. You must still separate
related documents, users, and paraphrases according to your application.
Each epoch reports mean training cross entropy and, when supplied, validation
cross entropy. The last requested epoch is saved; validation does not select a
checkpoint or alter training.

For a Hugging Face dataset, specify the split and a JSON mapping from Jeva field
names to the dataset's top-level columns:

```bash
python -m jeva.finetune \
  --model ./models/Jeva-4B \
  --dataset YOUR_ACCOUNT/YOUR_DATASET \
  --train-split train --validation-split validation \
  --column-map '{"state":"text","question":"instruction","options":"candidates","target_index":"label"}' \
  --output-dir ./outputs/my-hf-jeva --mode head --device cuda
```

The default decision type is `choice`. To mix types, map `decision_type` to a
dataset column too. Mapping does not execute dataset-specific label conversions:
convert nested structures or string labels to the documented schema beforehand.
`--dataset-name` selects a dataset configuration when needed.

## Portable output and verified resume

The output contains `config.yaml`, `head.pt`, `tokenizer/`, any
`backbone_adapter/`, and a frozen copy of the base encoder in `backbone/`.
A base encoder specified by a Hub repository is snapshotted using its loaded
commit revision when available. Saved component paths
are relative to the output, and the original local source path is not retained.
Model heads and optimizer state are loaded with PyTorch's restricted
`weights_only=True` reader; adapters and copied backbone weights use
safetensors. Treat an output as complete only after `training.json` appears.

Move the entire directory and load it like any released model:

```python
from jeva import Jeva

model = Jeva.from_pretrained("./outputs/my-jeva", device="cuda")
```

`training.json` records data digests, settings, parameter names, losses, and file
checksums without recording dataset paths or example text. `trainer_state.pt`
stores the optimizer and PyTorch random-generator states. To continue a completed
run, supply the same ordered data, validation data, mode, device, and optimization
settings, increase the total epoch count, and choose another new output directory:

```bash
python -m jeva.finetune \
  --resume-from ./outputs/my-jeva \
  --train-data ./my_train.jsonl --validation-data ./my_validation.jsonl \
  --output-dir ./outputs/my-jeva-continued \
  --mode head --device cuda \
  --epochs 5 --batch-size 1 --gradient-accumulation 8 \
  --learning-rate 0.0001
```

Resume checks file hashes, data digests, parameter names, and settings before
restoring optimizer and random-generator state. GPU kernels can still vary
across hardware and library versions. To start a new optimization run from a
fine-tuned checkpoint, use `--model` instead of `--resume-from`.
