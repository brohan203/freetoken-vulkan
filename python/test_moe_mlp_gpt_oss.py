"""test_moe_mlp_gpt_oss.py — verify gpt-oss MoE MLP shader against the
transformers reference on ACTUAL gpt-oss-20b layer-0 weights.

This is the largest correctness milestone yet — full MoE MLP forward using
MXFP4 experts, interleaved gate/up, and gpt-oss's specific activation.
"""
from __future__ import annotations
import os, pathlib, sys, time, json, math

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.cpp_extension import load

HERE  = pathlib.Path(__file__).resolve().parent
REPO  = HERE.parent
PYDIR = REPO / "python"
sys.path.insert(0, str(PYDIR))
from gpt_oss.mxfp4_ref import mxfp4_dequant

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

# ---- Load real gpt-oss weights (layer 0) ----
MODEL = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b")
with open(MODEL / "model.safetensors.index.json") as f:
    idx = json.load(f)["weight_map"]

def get(name):
    with safe_open(MODEL / idx[name], framework="pt") as f:
        return f.get_tensor(name)

L = 0
gu_blocks = get(f"model.layers.{L}.mlp.experts.gate_up_proj_blocks")   # [32, 5760, 90, 16] u8
gu_scales = get(f"model.layers.{L}.mlp.experts.gate_up_proj_scales")   # [32, 5760, 90]     u8
gu_bias   = get(f"model.layers.{L}.mlp.experts.gate_up_proj_bias")     # [32, 5760]         bf16
d_blocks  = get(f"model.layers.{L}.mlp.experts.down_proj_blocks")      # [32, 2880, 90, 16] u8
d_scales  = get(f"model.layers.{L}.mlp.experts.down_proj_scales")      # [32, 2880, 90]     u8
d_bias    = get(f"model.layers.{L}.mlp.experts.down_proj_bias")        # [32, 2880]         bf16

# Biases come as bf16; upgrade to fp32 for our fp32 shader
gu_bias_f32 = gu_bias.float().contiguous()
d_bias_f32  = d_bias.float().contiguous()

E, TWO_DFF, NB_D, PACK = gu_blocks.shape
D = 2880   # from config
Dff = TWO_DFF // 2
K = 4
print(f"Layer {L}: E={E}  D={D}  Dff={Dff}  2*Dff={TWO_DFF}")
print(f"gu_blocks: {list(gu_blocks.shape)} {gu_blocks.dtype}")
print(f"gu_bias:   {list(gu_bias.shape)} {gu_bias.dtype} -> fp32")
print(f"d_bias:    {list(d_bias.shape)} {d_bias.dtype} -> fp32")


# ---- Reference: pure-PyTorch forward matching transformers.GptOssExperts ----
def gpt_oss_moe_ref(x, indices, routing_weights,
                    W_gu_deq, gu_bias, W_d_deq, d_bias,
                    alpha=1.702, limit=7.0):
    """
    x:               [T, D] fp32
    indices:         [T, K] int
    routing_weights: [T, K] fp32
    W_gu_deq:        [E, 2*Dff, D] fp32  (already dequantized)
    gu_bias:         [E, 2*Dff]    fp32
    W_d_deq:         [E, D, Dff]   fp32
    d_bias:          [E, D]        fp32
    Returns y: [T, D] fp32
    """
    T, D = x.shape
    K = indices.shape[1]
    y = torch.zeros(T, D, dtype=torch.float32)

    for t in range(T):
        for k in range(K):
            e = indices[t, k].item()
            w = routing_weights[t, k].item()

            gate_up = W_gu_deq[e] @ x[t] + gu_bias[e]        # [2*Dff]
            gate = gate_up[0::2]                              # even = gate
            up   = gate_up[1::2]                              # odd  = up

            gate = torch.clamp(gate, max=limit)
            up   = torch.clamp(up,   min=-limit, max=limit)
            glu  = gate * torch.sigmoid(gate * alpha)
            hidden = (up + 1.0) * glu                          # [Dff]

            out = W_d_deq[e] @ hidden + d_bias[e]              # [D]
            y[t] += w * out
    return y


# ---- Test at real gpt-oss geometry, small T for reference speed ----
# T=1 keeps ref computation fast (matmul is 5760x2880 per expert per token)
torch.manual_seed(0xC0FFEE)
T = 1
x = torch.randn(T, D) * 0.1

# Fake router: top-K over random logits
logits = torch.randn(T, E) * 2.0
probs  = torch.softmax(logits, dim=-1)
weights_r, indices = torch.topk(probs, K, dim=-1)
weights_r = weights_r / weights_r.sum(-1, keepdim=True)

# Dequantize expert weights ONCE for the reference (slow but correct)
print("\nDequantizing all 32 experts for reference (may take a minute)...")
t0 = time.time()
W_gu_deq = mxfp4_dequant(gu_blocks, gu_scales)   # [32, 5760, 2880] fp32 — ~2 GB!
W_d_deq  = mxfp4_dequant(d_blocks,  d_scales)    # [32, 2880, 2880] fp32
print(f"  dequant elapsed: {time.time()-t0:.1f}s")
print(f"  W_gu_deq: {list(W_gu_deq.shape)}  {W_gu_deq.nbytes/1024**2:.0f} MB")
print(f"  W_d_deq:  {list(W_d_deq.shape)}  {W_d_deq.nbytes/1024**2:.0f} MB")

print("\nRunning PyTorch reference (this is slow)...")
t0 = time.time()
y_ref = gpt_oss_moe_ref(x, indices, weights_r,
                        W_gu_deq, gu_bias_f32,
                        W_d_deq,  d_bias_f32)
print(f"  ref elapsed: {time.time()-t0:.1f}s")

print("\nRunning Vulkan shader...")
t0 = time.time()
y_gpu = ext.moe_mlp_gpt_oss(
    x, indices, weights_r,
    gu_blocks, gu_scales, gu_bias_f32,
    d_blocks,  d_scales,  d_bias_f32,
)
print(f"  gpu elapsed: {time.time()-t0:.1f}s")

max_abs = (y_gpu - y_ref).abs().max().item()
rel     = (y_gpu - y_ref).abs() / y_ref.abs().clamp_min(1e-6)
max_rel = rel.max().item()
print(f"\nmax|D|         = {max_abs:.3e}")
print(f"max|D/y|       = {max_rel:.3e}")
print(f"y_ref stats:   mean={y_ref.mean():.4f}  std={y_ref.std():.4f}  "
      f"range=[{y_ref.min():.3f}, {y_ref.max():.3f}]")
print(f"y_gpu stats:   mean={y_gpu.mean():.4f}  std={y_gpu.std():.4f}  "
      f"range=[{y_gpu.min():.3f}, {y_gpu.max():.3f}]")

ok = torch.allclose(y_gpu, y_ref, rtol=1e-3, atol=1e-3)
print(f"\n{'PASS' if ok else 'FAIL'}")
