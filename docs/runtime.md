# Runtime and installation

## Recommended NVIDIA environment

Create a clean conda environment with Python 3.12 on Linux x86_64, then install
`requirements.txt`. It pairs the CUDA 12.8 PyTorch 2.9.1 wheel with the matching
Python 3.12, CXX11-ABI causal convolution wheel. The optional linear-attention
and backward-kernel dependencies are pinned as well. Use a driver compatible
with this CUDA runtime. These instructions do not require an inference server.

```bash
conda create -n jeva -c nvidia/label/cuda-12.8.1 \
  python=3.12 cuda-nvcc=12.8.93 -y
conda activate jeva
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jeva-doctor --check-kernels --strict
```

Run `jeva-doctor --backward --strict` before LoRA adaptation. Imports alone are
not enough to test binary compatibility: the execution check runs a small
convolution and gated-delta-rule kernel, and optionally their gradients.
First use may JIT-compile kernels. No model weights are needed for this check.

The pinned Hopper/Triton combination uses TileLang for affected backward
kernels, which requires a usable `nvcc`, not just CUDA runtime libraries. The
conda command above includes NVIDIA's CUDA 12.8 compiler and its compatible
development dependencies. After activation, `nvcc --version` should report
12.8. Do not set `FLA_TILELANG=0` or `FLA_DISABLE_BACKEND_DISPATCH=1` for that
path. See [NVIDIA's versioned conda installation guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html#conda-installation).

The optimized Qwen3.5 path needs both causal convolution and linear attention.
Without them, Transformers can use a slower reference path. The environment
check reports this instead of treating a successful model import as proof of
acceleration. See the [Transformers Qwen3.5 implementation notes](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[FLA installation](https://github.com/fla-org/flash-linear-attention#installation),
and [causal-conv1d releases](https://github.com/Dao-AILab/causal-conv1d/releases).

Other operating systems, Python versions, Torch ABIs, or GPU architectures need
an appropriately matched kernel build. Do not install the supplied binary into
an arbitrary existing Torch environment. `requirements-core.txt` omits the
optional compiled kernels and is a useful correctness fallback, but will not
reproduce accelerated timings. No latency guarantee applies to all GPUs.

## GPU memory and batching

The default uses BF16 model weights and limits option encoding to 16 rows and
16,384 padded tokens per sub-batch. Reduce these for smaller GPUs:

```python
from jeva import Jeva
model = Jeva.from_pretrained(
    "./models/Jeva-4B", device="cuda",
    option_chunk_size=4,
    option_batch_tokens=4096,
    option_cache_max_bytes=128 * 1024**2,
    set_cache_max_bytes=128 * 1024**2,
)
```

A single text longer than the batch token budget still has to run as one row;
that budget is a batching bound, not a chunking or truncation rule. Candidate
count and individual-text length are independent resource dimensions.
For large-memory GPUs, increase the encoding batch limits after measuring peak
memory and verifying results. The paper's high-throughput ToolRet setting used
256 rows / 131,072 padded tokens; it is not a portable default.

Both caches have entry and tensor-byte caps. Transient tensors, the model, and
Python bookkeeping use additional memory; a cache byte limit is not a total
GPU-memory limit. Reusing a cache across changed weights or tokenizer settings
is invalid. Adaptation uses a separate model load and no inference cache.

## Reproducible comparisons

- Pin the checkpoint revision and record package versions with `jeva-doctor`.
- Specify cache mode, token limits, batching limits, and truncation policy.
- Warm up kernels separately from the measured stream.
- For a natural request stream, include its cold first requests. For primed
  memory, report priming cost separately; do not call it a cold request.
- Keep full-request response caching distinct from option-memory reuse.
- Measure latency only on a GPU allocated to your process and synchronize it.

The semantic symmetry is exact in real arithmetic. Different batch shapes and
kernel reductions can introduce small floating-point differences. Check
probabilities and decisions as well as timing when changing the runtime.

## Local offline loading

`jeva-download` fetches inference assets only. With an authorized Hub account,
download once, then use `local_files_only=True`. Keep `config.yaml`, `head.pt`,
the base encoder, the tokenizer, and `backbone_adapter/` together.
LoRA loading verifies the complete adapter key hierarchy and refuses partial
loads. Inference merging changes only the in-memory copy, not the files.

The model head uses the restricted PyTorch `weights_only=True` reader; encoder
and adapter weights use safetensors. `trust_remote_code` is false by default.
