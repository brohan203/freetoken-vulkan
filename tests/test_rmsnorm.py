"""test_rmsnorm.py — correctness harness for the Vulkan RMSNorm kernel.

Approach:
  1. Generate random x, w tensors on CPU.
  2. Compute reference y_ref = rms_norm(x, w, eps)  using PyTorch.
  3. Dump x, w to raw float32 binary files.
  4. Invoke build\Release\rmsnorm.exe to compute y_gpu on the 6800 XT.
  5. Load y_gpu back, compare to y_ref.

We can't require bit-exact match: our GPU tree reduction sums 256 partials
in a different order than PyTorch's reduction, and float sums aren't
associative. Correct tolerance target for FP32 RMSNorm at hidden dims we
care about is ~1e-5 relative.
"""
from __future__ import annotations
import os, sys, subprocess, tempfile, pathlib

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parent.parent
EXE  = REPO / "build" / "Release" / "rmsnorm.exe"
BUILD_DIR = REPO / "build"   # exe expects shaders/ relative to cwd

# Test cases spanning shapes we care about for real transformer inference.
# H (hidden dim) is what matters — the reduction happens along H. Typical
# open-source LLM hidden dims: Phi-3.5-MoE=4096, Qwen3-30B-A3B=2048, GPT-2=768.
CASES = [
    # (N, H, eps, label)
    (  1,  128, 1e-6, "tiny"),
    (  8, 2048, 1e-6, "qwen-hidden"),
    ( 16, 4096, 1e-6, "phi-hidden"),
    (128, 4096, 1e-6, "batch128"),
    (  4, 8192, 1e-6, "wide"),
    (  4,  257, 1e-6, "non-power-of-2"),  # exercise the H % WG != 0 path
]

REL_TOL = 1e-5   # relative tolerance
ABS_TOL = 1e-5   # absolute tolerance (guards near-zero values)


def rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm — same formula as our shader, computed in FP32."""
    x = x.float()
    w = w.float()
    ss = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ss + eps) * w


def run_case(N: int, H: int, eps: float, label: str) -> bool:
    torch.manual_seed(0xC0FFEE + N * 131 + H)
    x = torch.randn(N, H, dtype=torch.float32)
    w = torch.rand(H, dtype=torch.float32) * 0.5 + 0.75  # avoid 0-magnitude weights

    y_ref = rms_norm_ref(x, w, eps)

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        x_path = td / "x.bin"
        w_path = td / "w.bin"
        y_path = td / "y.bin"
        x_path.write_bytes(x.contiguous().numpy().tobytes())
        w_path.write_bytes(w.contiguous().numpy().tobytes())

        result = subprocess.run(
            [str(EXE), str(x_path), str(w_path), str(y_path),
             str(N), str(H), f"{eps:g}"],
            cwd=str(BUILD_DIR),
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  [{label}]  exe failed rc={result.returncode}")
            print("  stdout:", result.stdout)
            print("  stderr:", result.stderr)
            return False

        y_gpu = np.frombuffer(y_path.read_bytes(), dtype=np.float32).reshape(N, H)

    y_gpu_t = torch.from_numpy(y_gpu.copy())

    abs_diff = (y_gpu_t - y_ref).abs()
    rel_diff = abs_diff / y_ref.abs().clamp_min(1e-6)
    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    mean_abs = abs_diff.mean().item()

    ok = torch.allclose(y_gpu_t, y_ref, rtol=REL_TOL, atol=ABS_TOL)
    tag = "PASS" if ok else "FAIL"
    # Pull first line of GPU-side stdout (the "GPU dispatch + wait: X.XXX ms")
    ms_line = next((l for l in result.stdout.splitlines() if "dispatch" in l), "")
    print(f"  [{label:16s}]  N={N:4d} H={H:5d}  max|D|={max_abs:.2e}  "
          f"max|D/y|={max_rel:.2e}  mean|D|={mean_abs:.2e}  {tag}  {ms_line}")
    return ok


def main() -> int:
    if not EXE.exists():
        print(f"ERROR: {EXE} does not exist. Build first:")
        print(f"  cmake --build build --config Release")
        return 1
    print(f"exe:      {EXE}")
    print(f"tol:      rtol={REL_TOL:.0e}  atol={ABS_TOL:.0e}")
    print()

    all_ok = True
    for case in CASES:
        all_ok &= run_case(*case)

    print()
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
