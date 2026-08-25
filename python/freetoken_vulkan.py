"""freetoken_vulkan — Python wrapper around our Vulkan compute kernels.

This module exposes each shader as a Python function that takes and returns
`torch.Tensor`. It's the API layer the rest of a real inference stack would
use.

WARN IMPLEMENTATION NOTE — SUBPROCESS SHIM
========================================

Under the hood, every call spawns the CLI exe, writes inputs to disk, reads
outputs back, and copies through host memory. This is *slow* — ~100 ms
overhead per call regardless of tensor size.

That's fine for CORRECTNESS demos (which is why we built it). It is NOT
usable for real inference — a 30-layer Phi-3.5-MoE forward pass has ~500
kernel calls, that's a minute per token just from IPC.

The RIGHT integration is a C++ Torch extension that:
  1. Registers ops as `torch.ops.freetoken_vulkan.rmsnorm(x, w, eps)` etc.
  2. Keeps a persistent `VulkanContext` alive for the process lifetime.
  3. Allocates Vulkan-backed `torch.Tensor` via a custom device type,
     so weights don't round-trip through CPU on every kernel.
  4. Records descriptor set + pipeline objects once per shape family
     (cache them by hash of (shape, dtype, stride)).
  5. Batches consecutive ops into a single command buffer submission.

That's ~2-3 days of engineering. Tracked in docs/SKIPPED.md.

For now — the subprocess shim proves the interface + kernel correctness
end-to-end.
"""
from __future__ import annotations
import math
import os
import pathlib
import subprocess
import tempfile

import numpy as np
import torch

_REPO = pathlib.Path(__file__).resolve().parent.parent
_BUILD = _REPO / "build"
_EXE_DIR = _BUILD / "Release"
_SHADER_DIR = _BUILD / "shaders"

# ---- environment for subprocess (needs Vulkan SDK on PATH) ----
_VULKAN_SDK = os.environ.get("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
_ENV = os.environ.copy()
_ENV["VULKAN_SDK"] = _VULKAN_SDK
_ENV["PATH"] = _VULKAN_SDK + r"\Bin;" + _ENV.get("PATH", "")


def _run_kernel(exe: str, args: list[str], cwd: pathlib.Path = _BUILD) -> str:
    r = subprocess.run(
        [str(_EXE_DIR / exe)] + args,
        cwd=str(cwd),
        env=_ENV,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"{exe} failed (rc={r.returncode}):\n{r.stdout}\n{r.stderr}")
    return r.stdout


def _write_tensor(path: pathlib.Path, t: torch.Tensor) -> None:
    assert t.dtype == torch.float32, f"only fp32 supported, got {t.dtype}"
    path.write_bytes(t.detach().contiguous().cpu().numpy().tobytes())


def _read_tensor(path: pathlib.Path, shape: tuple, dtype=np.float32) -> torch.Tensor:
    arr = np.frombuffer(path.read_bytes(), dtype=dtype).reshape(shape)
    return torch.from_numpy(arr.copy())


# ============================================================
# Public API — one function per kernel.
# ============================================================

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm(x, weight, eps). x: [..., H], weight: [H]. Returns [..., H]."""
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    N, H = x2.shape
    assert weight.shape == (H,)
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        xp, wp, yp = td / "x.bin", td / "w.bin", td / "y.bin"
        _write_tensor(xp, x2)
        _write_tensor(wp, weight)
        _run_kernel("rmsnorm.exe", [str(xp), str(wp), str(yp),
                                    str(N), str(H), f"{eps:g}"])
        y = _read_tensor(yp, (N, H))
    return y.reshape(orig_shape)


def matmul(A: torch.Tensor, B: torch.Tensor,
           variant: str = "reg_tiled") -> torch.Tensor:
    """C = A @ B via one of our GEMM shaders.
    variant: 'naive' | 'tiled' | 'reg_tiled'
    """
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[0]
    M, K = A.shape
    _, N = B.shape
    tile_by_variant = {"naive": (16, 16), "tiled": (16, 16), "reg_tiled": (64, 64)}
    if variant not in tile_by_variant:
        raise ValueError(f"unknown gemm variant {variant!r}")
    tm, tn = tile_by_variant[variant]
    shader = _SHADER_DIR / f"gemm_{variant}_f32.comp.spv"
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        ap, bp, cp = td / "a.bin", td / "b.bin", td / "c.bin"
        _write_tensor(ap, A)
        _write_tensor(bp, B)
        _run_kernel("gemm.exe", [str(shader), str(ap), str(bp), str(cp),
                                 str(M), str(N), str(K), "1", str(tm), str(tn)])
        return _read_tensor(cp, (M, N))


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Rowwise softmax over the last dim."""
    if dim != -1 and dim != x.ndim - 1:
        raise NotImplementedError("only last-dim softmax supported currently")
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    N, D = x2.shape
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        xp, yp = td / "x.bin", td / "y.bin"
        _write_tensor(xp, x2)
        _run_kernel("softmax.exe", [str(xp), str(yp), str(N), str(D)])
        y = _read_tensor(yp, (N, D))
    return y.reshape(orig_shape)


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    scale: float | None = None) -> torch.Tensor:
    """FlashAttention v1 — Q, K, V: [S, D]. Returns [S, D]."""
    assert Q.shape == K.shape == V.shape
    S, D = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    shader = _SHADER_DIR / "flash_attention_f32.comp.spv"
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        qp, kp, vp, op = td / "q.bin", td / "k.bin", td / "v.bin", td / "o.bin"
        _write_tensor(qp, Q); _write_tensor(kp, K); _write_tensor(vp, V)
        _run_kernel("attention.exe",
                    [str(shader), str(qp), str(kp), str(vp), str(op),
                     str(S), str(D), f"{scale:.9g}"])
        return _read_tensor(op, (S, D))


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU: silu(gate) * up, elementwise."""
    assert gate.shape == up.shape
    shape = gate.shape
    N = int(gate.numel())
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        gp, up_, yp = td / "g.bin", td / "u.bin", td / "y.bin"
        _write_tensor(gp, gate); _write_tensor(up_, up)
        _run_kernel("swiglu.exe",
                    [str(gp), str(up_), str(yp), str(N)])
        return _read_tensor(yp, shape)


def moe_router(logits: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-K softmax routing. logits: [T, E].
    Returns (indices [T, K] int64, weights [T, K] float32)."""
    T, E = logits.shape
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        lp, ip, wp = td / "l.bin", td / "i.bin", td / "w.bin"
        _write_tensor(lp, logits)
        _run_kernel("moe_router.exe",
                    [str(lp), str(ip), str(wp), str(T), str(E), str(K)])
        indices = _read_tensor(ip, (T, K), dtype=np.uint32).to(torch.int64)
        weights = _read_tensor(wp, (T, K))
    return indices, weights


def moe_mlp(x: torch.Tensor,
            indices: torch.Tensor, weights: torch.Tensor,
            W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor
            ) -> torch.Tensor:
    """Fused MoE MLP forward: routed SwiGLU MLP with K experts per token.

    Shapes:
      x:       [T, D]
      indices: [T, K]  (int64 or int32 — cast internally to uint32 for the shader)
      weights: [T, K]
      W_gate:  [E, Dff, D]
      W_up:    [E, Dff, D]
      W_down:  [E, D, Dff]
    Returns y: [T, D]

    Computes per-token:
      y[t] = sum_k weights[t,k] * W_down[e_k] @ (silu(W_gate[e_k] @ x[t])
                                                 * (W_up[e_k] @ x[t]))
    where e_k = indices[t, k].

    Current shader limits: D ≤ 256, Dff ≤ 512 (LDS budget).
    """
    T, D = x.shape
    _, K = indices.shape
    E, Dff, _ = W_gate.shape
    assert W_gate.shape == (E, Dff, D)
    assert W_up.shape == (E, Dff, D)
    assert W_down.shape == (E, D, Dff)
    assert weights.shape == (T, K)
    if D > 256 or Dff > 512:
        raise ValueError(f"D={D} Dff={Dff} exceeds shader MAX_D=256, MAX_DFF=512")

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        x_p  = td / "x.bin";  i_p  = td / "i.bin";  w_p  = td / "w.bin"
        wg_p = td / "wg.bin"; wu_p = td / "wu.bin"; wd_p = td / "wd.bin"
        y_p  = td / "y.bin"

        _write_tensor(x_p, x)
        i_p.write_bytes(indices.to(torch.uint32).contiguous().cpu().numpy().tobytes())
        _write_tensor(w_p, weights)
        _write_tensor(wg_p, W_gate)
        _write_tensor(wu_p, W_up)
        _write_tensor(wd_p, W_down)

        _run_kernel("moe_mlp.exe",
                    [str(x_p), str(i_p), str(w_p),
                     str(wg_p), str(wu_p), str(wd_p), str(y_p),
                     str(T), str(D), str(Dff), str(E), str(K)])
        return _read_tensor(y_p, (T, D))


# ============================================================
# Sanity check (run `python -m freetoken_vulkan` to smoke-test).
# ============================================================
if __name__ == "__main__":
    print("freetoken_vulkan smoke test")
    torch.manual_seed(0)

    x = torch.randn(4, 128)
    w = torch.rand(128) + 0.5
    y = rmsnorm(x, w)
    y_ref = torch.nn.functional.rms_norm(x, (128,), w, eps=1e-6)
    print(f"  rmsnorm  max|D| = {(y - y_ref).abs().max():.2e}")

    A = torch.randn(64, 128); B = torch.randn(128, 32)
    C = matmul(A, B)
    print(f"  matmul   max|D| = {(C - A @ B).abs().max():.2e}")

    s = softmax(torch.randn(4, 128))
    print(f"  softmax  rows sum to 1 within {(s.sum(-1) - 1).abs().max():.2e}")

    print("  freetoken_vulkan wrapper OK")
