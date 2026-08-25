"""test_flash_attention_gpt_oss.py — verify GQA + causal + SWA + sinks
against pure-PyTorch reference matching transformers.gpt_oss semantics.
"""
from __future__ import annotations
import os, pathlib, sys, math, time

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load

HERE  = pathlib.Path(__file__).resolve().parent
REPO  = HERE.parent
PYDIR = REPO / "python"
vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])

print("Loading extension...")
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)


def gpt_oss_attn_ref(Q, K, V, sinks, scale, causal=True,
                    sliding_window=0, use_sinks=True):
    """Matches transformers.gpt_oss eager_attention_forward closely.
       Q: [B, H_q,  S, D]
       K: [B, H_kv, S, D]
       V: [B, H_kv, S, D]
       sinks: [H_q]
    """
    B, H_q, S, D = Q.shape
    _, H_kv, _, _ = K.shape
    n_rep = H_q // H_kv

    # Repeat K, V along head dim to match Q's H_q.
    Kr = K.repeat_interleave(n_rep, dim=1)      # [B, H_q, S, D]
    Vr = V.repeat_interleave(n_rep, dim=1)

    scores = torch.einsum('bhsd,bhtd->bhst', Q, Kr) * scale  # [B, H_q, S_q, S_kv]

    # Causal mask.
    if causal:
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    # Sliding window (positions kv_pos < q_idx - window + 1 are masked).
    if sliding_window > 0:
        for q_idx in range(S):
            lo = max(0, q_idx - sliding_window + 1)
            scores[..., q_idx, :lo] = float('-inf')

    if use_sinks:
        sink_col = sinks.reshape(1, H_q, 1, 1).expand(B, H_q, S, 1)  # [B, H_q, S, 1]
        combined = torch.cat([scores, sink_col], dim=-1)              # [..., S+1]
        probs = torch.softmax(combined, dim=-1)
        probs = probs[..., :-1]                                        # drop sink col
    else:
        probs = torch.softmax(scores, dim=-1)

    out = torch.einsum('bhst,bhtd->bhsd', probs, Vr)   # [B, H_q, S, D]
    return out


# Shapes: gpt-oss-20b uses (H_q=64, H_kv=8, D=64, S≤131072)
# Small S for the reference to run fast.
CASES = [
    # (B, H_q, H_kv, S, D, sliding_window, use_sinks, label)
    ( 1,  8,  1,   64,  64,   0, False, "H_q=8 H_kv=1 GQA=8, no mask no sinks"),
    ( 1,  8,  1,   64,  64,   0, True,  "GQA + sinks (no SWA)"),
    ( 1,  8,  1,  128,  64, 128, False, "GQA + causal-only (window=128=S)"),
    ( 1,  8,  1,  256,  64,  64, False, "GQA + SWA=64"),
    ( 1,  8,  1,  256,  64,  64, True,  "GQA + SWA=64 + sinks (real gpt-oss cfg)"),
    ( 1, 64,  8,  128,  64,   0, True,  "REAL gpt-oss H_q=64 H_kv=8 (full attn)"),
    ( 1, 64,  8,  128,  64, 128, True,  "REAL gpt-oss (sliding=128 window)"),
    ( 2,  8,  1,   64,  64,   0, True,  "B=2 batch"),
]


def run(B, H_q, H_kv, S, D, sw, use_sinks, label):
    torch.manual_seed(0xAAA + S * 131 + H_q)
    scale = 1.0 / math.sqrt(D)
    Q = torch.randn(B, H_q,  S, D) * 0.1
    K = torch.randn(B, H_kv, S, D) * 0.1
    V = torch.randn(B, H_kv, S, D) * 0.1
    sinks = torch.randn(H_q) * 0.5

    y_ref = gpt_oss_attn_ref(Q, K, V, sinks, scale,
                              causal=True, sliding_window=sw, use_sinks=use_sinks)
    y_gpu = ext.flash_attention_gpt_oss(Q, K, V, sinks, scale, sw, use_sinks)

    diff = (y_gpu - y_ref).abs()
    max_abs = diff.max().item()
    ok = torch.allclose(y_gpu, y_ref, rtol=1e-3, atol=1e-4)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{label:44s}]  B={B} H_q={H_q:2d} H_kv={H_kv} S={S:4d}  "
          f"max|D|={max_abs:.2e}  {tag}")
    return ok


all_ok = True
for c in CASES:
    all_ok &= run(*c)
print("\nALL PASS" if all_ok else "\nSOME FAILED")
