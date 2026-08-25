"""test_gemm.py — correctness + perf comparison for Vulkan GEMM variants.

We swap shaders (naive vs tiled vs ...) without rebuilding by passing the
SPV path to the exe. Each variant is validated against torch.matmul and
its GFLOPS is compared.

Tolerance notes: FP32 GEMM accumulates K products per output element. Errors
grow ~sqrt(K), so we widen tolerance vs. RMSNorm. For K=1024, expected
relative error is roughly sqrt(1024)*2^-23 ≈ 4e-6, so rtol=1e-4 is safe.
"""
from __future__ import annotations
import os, sys, subprocess, tempfile, pathlib

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
EXE  = REPO / "build" / "Release" / "gemm.exe"
BUILD_DIR = REPO / "build"

SHADERS = {
    # variant_name: (spv_path, tile_m, tile_n, dtype)
    "naive":     (BUILD_DIR / "shaders" / "gemm_naive_f32.comp.spv",     16, 16, "f32"),
    "tiled":     (BUILD_DIR / "shaders" / "gemm_tiled_f32.comp.spv",     16, 16, "f32"),
    "reg_tiled": (BUILD_DIR / "shaders" / "gemm_reg_tiled_f32.comp.spv", 64, 64, "f32"),
    "reg_tiled_f16": (BUILD_DIR / "shaders" / "gemm_reg_tiled_f16.comp.spv", 64, 64, "f16"),
}

# (M, N, K, label)
CASES = [
    (   16,   16,   16, "tiny"),
    (   64,   64,   64, "small"),
    (  256,  256,  256, "medium"),
    ( 1024, 1024, 1024, "1024-cubed"),
    (    1, 4096, 4096, "gemv (vector*matrix)"),
    (   17,   33,   19, "irregular"),
]

REL_TOL = 1e-4
ABS_TOL = 1e-4
# FP16 loses ~3 decimal digits and accumulates faster — much wider tol needed
REL_TOL_F16 = 5e-2
ABS_TOL_F16 = 5e-2
ITERS   = 5  # for timing


def run_case(shader_path: pathlib.Path, tile_m: int, tile_n: int, dtype: str,
             M: int, N: int, K: int, label: str) -> tuple[bool, str]:
    torch.manual_seed(0xBEEF + M * 131 + K)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    C_ref = A @ B

    if dtype == "f16":
        # Cast to fp16 for GPU; reference stays fp32 for the diff.
        # This tests that our mixed-precision (fp16 storage, fp32 accum)
        # kernel gives an answer close to full-fp32 matmul.
        A_bin = A.to(torch.float16).contiguous()
        B_bin = B.to(torch.float16).contiguous()
        rtol, atol = REL_TOL_F16, ABS_TOL_F16
        c_dtype = np.float16
    else:
        A_bin = A.contiguous()
        B_bin = B.contiguous()
        rtol, atol = REL_TOL, ABS_TOL
        c_dtype = np.float32

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        a_path = td / "a.bin"; b_path = td / "b.bin"; c_path = td / "c.bin"
        a_path.write_bytes(A_bin.numpy().tobytes())
        b_path.write_bytes(B_bin.numpy().tobytes())

        result = subprocess.run(
            [str(EXE), str(shader_path), str(a_path), str(b_path), str(c_path),
             str(M), str(N), str(K), str(ITERS), str(tile_m), str(tile_n)],
            cwd=str(BUILD_DIR), capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return False, f"exe rc={result.returncode}: {result.stderr[:400]}"

        C_gpu = np.frombuffer(c_path.read_bytes(), dtype=c_dtype).reshape(M, N)
        C_gpu = C_gpu.astype(np.float32)  # promote for comparison

    C_gpu_t = torch.from_numpy(C_gpu.copy())
    abs_diff = (C_gpu_t - C_ref).abs()
    rel_diff = abs_diff / C_ref.abs().clamp_min(1e-6)
    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()

    ok = torch.allclose(C_gpu_t, C_ref, rtol=rtol, atol=atol)
    perf_line = next((l for l in result.stdout.splitlines()
                     if "GPU:" in l and "GFLOPS" in l), "")
    if not perf_line:
        perf_line = next((l for l in result.stdout.splitlines()
                         if "GFLOPS" in l), "")
    tag = "PASS" if ok else "FAIL"
    return ok, (
        f"  [{label:24s}]  M={M:4d} N={N:4d} K={K:4d}  "
        f"max|D|={max_abs:.2e}  max|D/C|={max_rel:.2e}  {tag}  {perf_line}"
    )


def main() -> int:
    if not EXE.exists():
        print(f"ERROR: {EXE} does not exist. Build first.")
        return 1

    all_ok = True
    for variant, (shader, tm, tn, dtype) in SHADERS.items():
        if not shader.exists():
            print(f"skipping {variant}: SPV not found ({shader})")
            continue
        print(f"=== variant: {variant}  (tile={tm}x{tn}  dtype={dtype}) ===")
        print(f"    shader: {shader}")
        for case in CASES:
            ok, line = run_case(shader, tm, tn, dtype, *case)
            print(line)
            all_ok &= ok
        print()

    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
