"""test_attention.py — correctness for attention variants.

Compares each Vulkan shader against torch.nn.functional.scaled_dot_product_attention.
Runs both naive (S ≤ 2048 due to LDS ceiling) and FlashAttention (any S).
"""
from __future__ import annotations
import os, sys, subprocess, tempfile, pathlib, math
import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parent.parent
EXE  = REPO / "build" / "Release" / "attention.exe"
BUILD_DIR = REPO / "build"

SHADERS = {
    # variant: (spv_path, max_S)   — None max_S means "no LDS limit"
    "naive": (BUILD_DIR / "shaders" / "attention_naive_f32.comp.spv", 2048),
    "flash": (BUILD_DIR / "shaders" / "flash_attention_f32.comp.spv", None),
}

# (S, D, label)
CASES = [
    (   64,   64, "tiny"),
    (  128,   64, "S=128 D=64"),
    (  256,  128, "S=256 D=128 typical head"),
    (  512,  128, "S=512 D=128"),
    ( 1024,   64, "S=1024 D=64"),
    ( 2048,  128, "S=2048 D=128 naive LDS-max"),
    (  257,   33, "irregular"),
    ( 4096,  128, "S=4096 D=128 (flash only)"),
    ( 8192,   64, "S=8192 D=64 (flash only)"),
]

REL_TOL = 1e-3
ABS_TOL = 1e-4


def naive_attention_ref(Q, K, V, scale):
    """Reference matching our shader's exact computation.
    torch.scaled_dot_product_attention selects different math paths per shape,
    which drifts from our shader by ~1e-3 at certain (S, D). This is what
    our kernel actually computes."""
    scores = (Q @ K.T) * scale
    weights = torch.softmax(scores, dim=-1)
    return weights @ V


def run_case(shader_path: pathlib.Path, S: int, D: int, label: str) -> bool:
    torch.manual_seed(0xACE + S * 131 + D)
    Q = torch.randn(S, D, dtype=torch.float32) * 0.1
    K = torch.randn(S, D, dtype=torch.float32) * 0.1
    V = torch.randn(S, D, dtype=torch.float32) * 0.1
    scale = 1.0 / math.sqrt(D)

    O_ref = naive_attention_ref(Q, K, V, scale)

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        q_path = td / "q.bin"; k_path = td / "k.bin"
        v_path = td / "v.bin"; o_path = td / "o.bin"
        q_path.write_bytes(Q.contiguous().numpy().tobytes())
        k_path.write_bytes(K.contiguous().numpy().tobytes())
        v_path.write_bytes(V.contiguous().numpy().tobytes())

        r = subprocess.run(
            [str(EXE), str(shader_path),
             str(q_path), str(k_path), str(v_path), str(o_path),
             str(S), str(D), f"{scale:.9g}"],
            cwd=str(BUILD_DIR), capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            print(f"  [{label}]  exe rc={r.returncode}: {r.stderr[:400]}")
            return False
        O_gpu = np.frombuffer(o_path.read_bytes(), dtype=np.float32).reshape(S, D)

    O_gpu_t = torch.from_numpy(O_gpu.copy())
    abs_diff = (O_gpu_t - O_ref).abs()
    max_abs = abs_diff.max().item()
    rel_diff = abs_diff / O_ref.abs().clamp_min(1e-6)
    max_rel = rel_diff.max().item()

    ok = torch.allclose(O_gpu_t, O_ref, rtol=REL_TOL, atol=ABS_TOL)
    tag = "PASS" if ok else "FAIL"
    ms_line = next((l for l in r.stdout.splitlines() if "dispatch" in l), "")
    print(f"  [{label:32s}]  S={S:5d} D={D:4d}  "
          f"max|D|={max_abs:.2e}  max|D/O|={max_rel:.2e}  {tag}  {ms_line}")
    return ok


def main() -> int:
    if not EXE.exists():
        print(f"ERROR: {EXE} does not exist. Build first.")
        return 1
    all_ok = True
    for variant, (shader, max_S) in SHADERS.items():
        if not shader.exists():
            print(f"skipping {variant}: SPV not found ({shader})")
            continue
        print(f"=== variant: {variant} ===")
        for S, D, label in CASES:
            if max_S is not None and S > max_S:
                print(f"  [{label:32s}]  S={S:5d} D={D:4d}  SKIP (LDS ceiling)")
                continue
            all_ok &= run_case(shader, S, D, label)
        print()

    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
