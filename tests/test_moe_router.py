"""test_moe_router.py — top-K router correctness.

Verifies:
  1. Indices match torch.topk on softmax(logits) — modulo tie-breaking
  2. Weights are correctly renormalized (sum to 1 per token, within FP32 tol)
  3. Weight values match the corresponding renormalized softmax probabilities
"""
from __future__ import annotations
import os, sys, subprocess, tempfile, pathlib
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
EXE  = REPO / "build" / "Release" / "moe_router.exe"
BUILD_DIR = REPO / "build"

# (T, E, K, label)
CASES = [
    (   1,   8, 2, "1 tok, mixtral 8/2"),
    ( 128,  16, 2, "128 tok, phi 16/2"),
    (1024,  16, 2, "1024 tok, phi-scale"),
    (  32,   8, 1, "top-1"),
    (  32,  64, 8, "big E, K=8"),
    (   4, 128, 2, "MAX_E=128"),
]


def run_case(T: int, E: int, K: int, label: str) -> bool:
    torch.manual_seed(0xD00D + T * 131 + E * 17 + K)
    logits = torch.randn(T, E, dtype=torch.float32) * 2.0

    # Reference: softmax then top-K then renorm.
    probs = torch.softmax(logits, dim=-1)
    ref_weights, ref_indices = torch.topk(probs, K, dim=-1)  # [T, K]
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        l_path = td / "l.bin"; i_path = td / "i.bin"; w_path = td / "w.bin"
        l_path.write_bytes(logits.contiguous().numpy().tobytes())
        r = subprocess.run(
            [str(EXE), str(l_path), str(i_path), str(w_path),
             str(T), str(E), str(K)],
            cwd=str(BUILD_DIR), capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            print(f"  [{label}]  exe rc={r.returncode}: {r.stderr[:400]}")
            return False
        idx_gpu = np.frombuffer(i_path.read_bytes(), dtype=np.uint32).reshape(T, K)
        wt_gpu  = np.frombuffer(w_path.read_bytes(), dtype=np.float32).reshape(T, K)

    idx_ref = ref_indices.numpy()
    wt_ref  = ref_weights.numpy()

    idx_match = (idx_gpu == idx_ref).all()
    wt_diff = np.abs(wt_gpu - wt_ref).max()
    row_sum_err = np.abs(wt_gpu.sum(-1) - 1.0).max()

    ok = idx_match and (wt_diff < 1e-5) and (row_sum_err < 1e-4)
    tag = "PASS" if ok else "FAIL"
    ms_line = next((l for l in r.stdout.splitlines() if "dispatch" in l), "")
    print(f"  [{label:28s}]  T={T:5d} E={E:3d} K={K}  "
          f"idx_ok={idx_match}  wt_diff={wt_diff:.2e}  sum-1={row_sum_err:.2e}  "
          f"{tag}  {ms_line}")
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
