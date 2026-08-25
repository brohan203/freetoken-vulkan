"""test_moe_mlp.py — end-to-end MoE MLP correctness against a pure-PyTorch reference.

Given x, indices, weights, and per-expert gate/up/down weight tensors,
compute:
    y[t] = sum_k weights[t,k] * W_down[e_k] @ (silu(W_gate[e_k] @ x[t]) *
                                                (W_up[e_k] @ x[t]))
where e_k = indices[t, k].

Reference computed in pure PyTorch. GPU output must match to accumulated
FP32 tolerance across ~4 matmuls per expert per token.
"""
from __future__ import annotations
import os, sys, subprocess, tempfile, pathlib
import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parent.parent
EXE  = REPO / "build" / "Release" / "moe_mlp.exe"
BUILD_DIR = REPO / "build"

# (T, D, Dff, E, K, label)   — kept small so LDS budgets fit our shader's
# MAX_D=256, MAX_DFF=512, and so the test runs fast under subprocess IPC.
CASES = [
    (   4,  32,  64,  4, 2, "tiny E=4 K=2"),
    (  16,  64, 128,  8, 2, "small phi-ish shape"),
    (  32, 128, 256,  8, 2, "medium"),
    (  64, 128, 256, 16, 2, "16 experts (phi count)"),
    (   8, 128, 256,  8, 1, "K=1 (single-expert)"),
    (   4, 256, 512,  4, 2, "MAX_D and MAX_DFF"),
    (  16,  64, 128,  8, 4, "K=4"),
]


def moe_mlp_ref(x, indices, weights, W_gate, W_up, W_down):
    """
    x:        [T, D]
    indices:  [T, K]   (int64)
    weights:  [T, K]   (float)
    W_gate:   [E, Dff, D]
    W_up:     [E, Dff, D]
    W_down:   [E, D, Dff]
    returns:  [T, D]
    """
    T, D = x.shape
    K = indices.shape[1]
    y = torch.zeros(T, D, dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            e = indices[t, k].item()
            w = weights[t, k].item()
            g = W_gate[e] @ x[t]         # [Dff]
            u = W_up[e]   @ x[t]         # [Dff]
            m = F.silu(g) * u             # [Dff]
            o = W_down[e] @ m             # [D]
            y[t] += w * o
    return y


def run_case(T: int, D: int, Dff: int, E: int, K: int, label: str) -> bool:
    torch.manual_seed(0xE001 + T * 131 + D * 17 + E)

    # Small-magnitude to keep numerics well-conditioned
    x = torch.randn(T, D) * 0.1
    W_gate = torch.randn(E, Dff, D) * (1.0 / (D ** 0.5))
    W_up   = torch.randn(E, Dff, D) * (1.0 / (D ** 0.5))
    W_down = torch.randn(E, D, Dff) * (1.0 / (Dff ** 0.5))

    # Fake router output: pick top-K expert IDs and weights that sum to 1
    logits = torch.randn(T, E) * 2.0
    probs  = torch.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, K, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # renormalize

    # Reference
    y_ref = moe_mlp_ref(x, indices, weights, W_gate, W_up, W_down)

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        x_p  = td / "x.bin";  i_p  = td / "i.bin";  w_p  = td / "w.bin"
        wg_p = td / "wg.bin"; wu_p = td / "wu.bin"; wd_p = td / "wd.bin"
        y_p  = td / "y.bin"

        x_p.write_bytes(x.contiguous().numpy().tobytes())
        i_p.write_bytes(indices.to(torch.uint32).contiguous().numpy().tobytes())
        w_p.write_bytes(weights.contiguous().numpy().tobytes())
        wg_p.write_bytes(W_gate.contiguous().numpy().tobytes())
        wu_p.write_bytes(W_up.contiguous().numpy().tobytes())
        wd_p.write_bytes(W_down.contiguous().numpy().tobytes())

        r = subprocess.run(
            [str(EXE), str(x_p), str(i_p), str(w_p),
             str(wg_p), str(wu_p), str(wd_p), str(y_p),
             str(T), str(D), str(Dff), str(E), str(K)],
            cwd=str(BUILD_DIR), capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            print(f"  [{label}]  exe rc={r.returncode}: {r.stderr[:400]}")
            return False
        y_gpu = np.frombuffer(y_p.read_bytes(), dtype=np.float32).reshape(T, D)

    y_gpu_t = torch.from_numpy(y_gpu.copy())
    abs_diff = (y_gpu_t - y_ref).abs()
    max_abs = abs_diff.max().item()

    # 3 matmul reductions per expert with 128-4096 accumulations → wider tol
    ok = torch.allclose(y_gpu_t, y_ref, rtol=1e-3, atol=1e-4)
    tag = "PASS" if ok else "FAIL"
    ms_line = next((l for l in r.stdout.splitlines() if "dispatch" in l), "")
    print(f"  [{label:28s}]  T={T:3d} D={D:3d} Dff={Dff:3d} E={E:2d} K={K}  "
          f"max|D|={max_abs:.2e}  {tag}  {ms_line}")
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
