from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys


def inspect(*, check_kernels=False, backward=False):
    report = {"python": platform.python_version(), "platform": platform.platform(), "packages": {}}
    issues = []
    for package in ("torch", "transformers", "peft", "causal-conv1d", "fla-core", "tilelang"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    if report["packages"]["peft"] is None:
        report["peft_import"] = False
        issues.append("PEFT is not installed; the released checkpoint requires its LoRA adapter loader.")
    else:
        try:
            importlib.import_module("peft")
            report["peft_import"] = True
        except Exception as exc:
            report["peft_import"] = False
            issues.append(f"PEFT import failed: {type(exc).__name__}: {exc}")
    import torch
    report["cuda_available"] = torch.cuda.is_available()
    report["torch_cuda"] = torch.version.cuda
    report["torch_cxx11_abi"] = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    if sys.version_info[:2] != (3, 12):
        issues.append("The pinned installation supports Python 3.12.")
    if (report["packages"]["torch"] or "").split("+")[0] != "2.9.1":
        issues.append("The supplied causal-conv1d binary targets torch 2.9.x; use the pinned environment.")
    if not report["cuda_available"]:
        issues.append("No visible NVIDIA GPU. CPU-only checks cannot validate the GPU runtime.")
    else:
        report["gpu"] = torch.cuda.get_device_name()
        report["compute_capability"] = torch.cuda.get_device_capability()
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
        if not report["bf16_supported"]:
            issues.append("The release uses BF16; this GPU does not report BF16 support.")
    try:
        module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        report["qwen35_import"] = True

        report["qwen35_fast_path"] = getattr(module, "is_fast_path_available", None)
        if report["qwen35_fast_path"] is False:
            issues.append("Transformers reports the Qwen3.5 fast path is unavailable; optimized imports alone are insufficient.")
    except Exception as exc:
        report["qwen35_import"] = False
        issues.append(f"Qwen3.5 import failed: {type(exc).__name__}: {exc}")
    try:
        from causal_conv1d import causal_conv1d_fn
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        report["fast_kernel_imports"] = True
    except Exception as exc:
        report["fast_kernel_imports"] = False
        issues.append(f"Optimized kernel import failed: {type(exc).__name__}: {exc}")
    if backward and report["cuda_available"] and report["fast_kernel_imports"]:
        try:


            utils = importlib.import_module("fla.utils")
            backend = importlib.import_module("fla.ops.common.backends.tilelang").TileLangBackend
            required = bool(utils.IS_NVIDIA_HOPPER and utils.TRITON_ABOVE_3_4_0
                            and not utils.TRITON_ABOVE_3_7_1)
            status = {"tilelang_required": required,
                      "tilelang_available_with_nvcc": bool(backend.is_available()),
                      "tilelang_enabled": bool(backend.is_enabled()),
                      "dispatch_disabled": os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1"}
            report["backward_backend"] = status
            if required and not status["tilelang_available_with_nvcc"]:
                issues.append("This Hopper/Triton combination requires TileLang and a usable CUDA nvcc compiler for backward.")
            if required and (not status["tilelang_enabled"] or status["dispatch_disabled"]):
                issues.append("The required FLA TileLang backward backend is disabled; check FLA_TILELANG and FLA_DISABLE_BACKEND_DISPATCH.")
        except Exception as exc:
            issues.append(f"FLA backward backend inspection failed: {type(exc).__name__}: {exc}")
    if check_kernels and report["cuda_available"] and report["fast_kernel_imports"]:
        try:

            x = torch.randn(1, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=backward)
            w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16, requires_grad=backward)
            y = causal_conv1d_fn(x, w, activation="silu")
            shape = (1, 64, 2, 64)
            q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=backward) for _ in range(3)]
            g = -torch.rand(1, 64, 2, device="cuda", dtype=torch.float32)
            beta = torch.sigmoid(torch.randn(1, 64, 2, device="cuda", dtype=torch.bfloat16))
            out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
            if backward:
                (out.float().square().mean() + y.float().square().mean()).backward()
                if not all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v, x, w)):
                    raise RuntimeError("Non-finite or missing test gradient")
            torch.cuda.synchronize()
            if not torch.isfinite(out).all() or not torch.isfinite(y).all():
                raise RuntimeError("Non-finite test output")
            report["kernel_execution"] = "forward+backward passed" if backward else "forward passed"
        except Exception as exc:
            report["kernel_execution"] = "failed"
            issues.append(f"Kernel execution failed: {type(exc).__name__}: {exc}")
    report["issues"] = issues
    return report


def main():
    parser = argparse.ArgumentParser(description="Check the Jeva installation and optional CUDA kernels")
    parser.add_argument("--check-kernels", action="store_true")
    parser.add_argument("--backward", action="store_true", help="Also verify backward kernels; implies --check-kernels")
    parser.add_argument("--strict", action="store_true", help="Exit unsuccessfully if any compatibility issue is detected")
    args = parser.parse_args()
    result = inspect(check_kernels=args.check_kernels or args.backward, backward=args.backward)
    print(json.dumps(result, indent=2, default=str))
    if args.strict and result["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
