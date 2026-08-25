"""test_softmax.py — correctness for rowwise softmax."""
from __future__ import annotations
import os, sys, subprocess, tempfile, pathlib
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
EXE  = REPO / "build" / "Release" / "softmax.exe"
BUILD_DIR = REPO / "build"

# (N, D, label, extreme?)  extreme=True adds large-magnitude inputs to test
# numerical stability
CASES = [
    (   1,  128, "tiny",       False),
    (   8, 2048, "qwen-hidden", False),
    (  16, 4096, "phi-hidden", False),
    (   4,  512, "attention-scores", False),
    (   4,  257, "non-power-of-2", False),
    (   4, 2048, "extreme-values (large -> tests -max stability)", True),
]

REL_TOL = 1e-4
ABS_TOL = 1e-5


def run_case(N: int, D: int, label: str, extreme: bool) -> bool:
    torch.manual_seed(0xF00D + N * 131 + D)
    if extreme:
        # Wide range including big positives — naive softmax would overflow.
        x = torch.randn(N, D) * 20.0 + 50.0
    else:
        x = torch.randn(N, D)

    y_ref = torch.softmax(x, dim=-1)

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        x_path = td / "x.bin"; y_path = td / "y.bin"
        x_path.write_bytes(x.contiguous().numpy().tobytes())
        r = subprocess.run(
            [str(EXE), str(x_path), str(y_path), str(N), str(D)],
            cwd=str(BUILD_DIR), capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            print(f"  [{label}]  exe rc={r.returncode}: {r.stderr[:400]}")
            return False
        y_gpu = np.frombuffer(y_path.read_bytes(), dtype=np.float32).reshape(N, D)

    y_gpu_t = torch.from_numpy(y_gpu.copy())
    abs_diff = (y_gpu_t - y_ref).abs()
    rel_diff = abs_diff / y_ref.abs().clamp_min(1e-6)
    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()

    # Sanity: rows should sum to 1
    row_sums = y_gpu_t.sum(dim=-1)
    sum_err = (row_sums - 1.0).abs().max().item()

    ok = torch.allclose(y_gpu_t, y_ref, rtol=REL_TOL, atol=ABS_TOL) and sum_err < 1e-3
    tag = "PASS" if ok else "FAIL"
    ms_line = next((l for l in r.stdout.splitlines() if "dispatch" in l), "")
    print(f"  [{label:44s}]  N={N:3d} D={D:4d}  "
          f"max|D|={max_abs:.2e}  rows_sum-1={sum_err:.2e}  {tag}  {ms_line}")
    return ok


def main() -> int:
    if not EXE.exists():
        print(f"ERROR: {EXE} does not exist. Build first.")
        return 1
    all_ok = True
    for c in CASES:
        all_ok &= run_case(*c)
    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
