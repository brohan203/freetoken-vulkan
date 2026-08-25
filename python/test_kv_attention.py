"""test_kv_attention.py — verify the KV-cache-aware attention shader.

Two invariants to check:

(A) Backward-compat: when past_len=0 and S_q=S_kv, the KV shader must
    produce IDENTICAL output to the non-KV shader. This confirms the
    algorithmic changes (indexing by S_q vs S_kv, past_len shift in
    causal mask) don't change the math.

(B) KV equivalence: running the KV shader in TWO steps (prefill on the
    first N-1 tokens, then decode on token N with the past K/V) must
    produce the SAME output as running the KV shader in ONE step (full
    N-token prefill). This is the property we exploit for fast decode.
"""
from __future__ import annotations
import os, pathlib, sys, math, time

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load

HERE  = pathlib.Path(__file__).resolve().parent
REPO  = HERE.parent
PYDIR = REPO / "python"
sys.path.insert(0, str(PYDIR))

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


def build(B, H_q, H_kv, S, D, sw, use_sinks, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(B, H_q,  S, D) * 0.1
    K = torch.randn(B, H_kv, S, D) * 0.1
    V = torch.randn(B, H_kv, S, D) * 0.1
    sinks = torch.randn(H_q) * 0.5
    scale = 1.0 / math.sqrt(D)
    return Q, K, V, sinks, scale


# ============================================================
# Invariant A: KV-shader matches non-KV shader when past_len=0, S_q=S_kv
# ============================================================
print("\n== Invariant A: KV shader with past_len=0 vs original shader ==")
CASES_A = [
    # (B, H_q, H_kv, S, D, sw, use_sinks, label)
    (1,  8,  1,   64,  64,   0, True,  "H_q=8 no SWA, sinks"),
    (1,  8,  1,  128,  64, 128, True,  "SWA=128"),
    (1, 64,  8,  128,  64,   0, True,  "real gpt-oss H_q=64 H_kv=8"),
    (1, 64,  8,  128,  64, 128, True,  "real gpt-oss + SWA=128"),
]
all_ok = True
for B, H_q, H_kv, S, D, sw, us, label in CASES_A:
    Q, K, V, sinks, scale = build(B, H_q, H_kv, S, D, sw, us, seed=0xA)

    y_old = ext.flash_attention_gpt_oss(Q, K, V, sinks, scale, sw, us)
    y_new = ext.flash_attention_gpt_oss_kv(Q, K, V, sinks, scale, 0, sw, us)

    diff = (y_new - y_old).abs().max().item()
    ok = torch.allclose(y_new, y_old, rtol=1e-5, atol=1e-6)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{label:40s}]  max|D|={diff:.2e}  {tag}")
    all_ok &= ok


# ============================================================
# Invariant B: incremental decode == full prefill
# For each S, run in two ways and diff:
#   full:  KV shader on Q[:S], K[:S], V[:S] with past_len=0    -> y_full [B, H, S, D]
#   inc:   KV shader on Q[:S-1] first (past_len=0, S_q=S-1),
#          then Q[-1:] on K[:S], V[:S] (past_len=S-1, S_q=1)   -> y_inc  [B, H, S, D]
# The last row (position S-1) must match between them.
# ============================================================
print("\n== Invariant B: incremental (prefill + decode) matches full prefill ==")
CASES_B = [
    (1,  8,  1,   32, 64,   0, True,  "small full-attn"),
    (1,  8,  1,   64, 64,  16, True,  "SWA=16"),
    (1, 64,  8,   64, 64,   0, True,  "real gpt-oss config"),
    (1, 64,  8,   64, 64, 128, True,  "real gpt-oss + SWA (window larger than S)"),
]
for B, H_q, H_kv, S, D, sw, us, label in CASES_B:
    Q, K, V, sinks, scale = build(B, H_q, H_kv, S, D, sw, us, seed=0xB)

    # Full prefill.
    y_full = ext.flash_attention_gpt_oss_kv(Q, K, V, sinks, scale, 0, sw, us)

    # Incremental: prefill S-1 tokens, then decode token S-1.
    y_pre  = ext.flash_attention_gpt_oss_kv(
        Q[:, :, :S-1, :].contiguous(),
        K[:, :, :S-1, :].contiguous(),
        V[:, :, :S-1, :].contiguous(),
        sinks, scale, past_len=0, sliding_window=sw, use_sinks=us,
    )
    y_dec  = ext.flash_attention_gpt_oss_kv(
        Q[:, :, S-1:S, :].contiguous(),
        K.contiguous(),      # full K/V (cache is what would be after append)
        V.contiguous(),
        sinks, scale, past_len=S-1, sliding_window=sw, use_sinks=us,
    )

    # Reconstruct full output from (prefill, decode). Diff against y_full.
    y_recon = torch.cat([y_pre, y_dec], dim=2)
    diff_all  = (y_recon - y_full).abs().max().item()
    diff_last = (y_dec[:, :, 0, :] - y_full[:, :, S-1, :]).abs().max().item()
    ok = torch.allclose(y_recon, y_full, rtol=1e-4, atol=1e-5)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{label:40s}]  max|D_all|={diff_all:.2e}  "
          f"max|D_lastQ|={diff_last:.2e}  {tag}")
    all_ok &= ok

print(f"\n{'ALL PASS' if all_ok else 'SOME FAILED'}")
