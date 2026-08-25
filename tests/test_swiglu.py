"""test_swiglu.py — elementwise SwiGLU correctness."""
from __future__ import annotations
import os, sys, subprocess, tempfile, pathlib
import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parent.parent
EXE  = REPO / "build" / "Release" / "swiglu.exe"
BUILD_DIR = REPO / "build"

# (shape, label)
CASES = [
    ((        256,), "small"),
    ((4 * 8192,   ), "large 1D"),
    ((16, 11008,  ), "phi-3.5 MoE ff dim"),
    ((257,        ), "non-power-of-2"),
    ((1_000_000,  ), "1M elements"),
]


def run_case(shape: tuple, label: str) -> bool:
    torch.manual_seed(0xF00D)
    N = int(np.prod(shape))
    gate = torch.randn(*shape, dtype=torch.float32)
    up   = torch.randn(*shape, dtype=torch.float32)
    y_ref = F.silu(gate) * up

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        g_path = td / "g.bin"; u_path = td / "u.bin"; y_path = td / "y.bin"
        g_path.write_bytes(gate.contiguous().numpy().tobytes())
        u_path.write_bytes(up.contiguous().numpy().tobytes())
        r = subprocess.run(
            [str(EXE), str(g_path), str(u_path), str(y_path), str(N)],
            cwd=str(BUILD_DIR), capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            print(f"  [{label}]  exe rc={r.returncode}: {r.stderr[:400]}")
            return False
        y_gpu = np.frombuffer(y_path.read_bytes(), dtype=np.float32).reshape(shape)

    y_gpu_t = torch.from_numpy(y_gpu.copy())
    abs_diff = (y_gpu_t - y_ref).abs()
    max_abs = abs_diff.max().item()
    ok = torch.allclose(y_gpu_t, y_ref, rtol=1e-5, atol=1e-6)
    tag = "PASS" if ok else "FAIL"
    ms_line = next((l for l in r.stdout.splitlines() if "dispatch" in l), "")
    print(f"  [{label:32s}]  N={N:>8d}  max|D|={max_abs:.2e}  {tag}  {ms_line}")
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
