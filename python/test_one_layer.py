"""test_one_layer.py — run one full gpt-oss-20b transformer layer through
our Vulkan port and compare against the transformers reference on the
same layer's weights.

This is the biggest correctness milestone yet — it composes ALL the
kernels we've built (rmsnorm on CPU, Q/K/V/O projections on CPU, RoPE on
GPU, FlashAttention with GQA + SWA + sinks on GPU, router on CPU, MoE
MLP with MXFP4 experts on GPU) into one transformer layer forward.

If this passes, we know we can string layers together and run the full
model. If it fails, we bisect: attention block only vs MoE block only.
"""
from __future__ import annotations
import os, pathlib, sys, time, math

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
from gpt_oss import load_model, gpt_oss_layer_forward
from gpt_oss.rope import compute_cos_sin_for_positions

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

MODEL_DIR = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b")
LAYER_IDX = 0

# ============ Load our weights ============
print(f"\nLoading layer {LAYER_IDX} weights (BF16 -> FP32 for non-MoE)...")
t0 = time.time()
model = load_model(MODEL_DIR, layers=[LAYER_IDX])
print(f"  loaded in {time.time()-t0:.1f}s")
cfg = model.config
lw = model.layers[0]

print(f"\ncfg: hidden={cfg.hidden_size}  H_q={cfg.num_attention_heads} H_kv={cfg.num_key_value_heads}  "
      f"head_dim={cfg.head_dim}  E={cfg.num_experts} top-K={cfg.num_experts_per_tok}")
print(f"     sliding_window={cfg.sliding_window}  "
      f"layer_type[{LAYER_IDX}]={cfg.layer_types[LAYER_IDX]}")

# ============ Reference: transformers eager layer forward ============
print("\nLoading transformers reference model (this reads the same safetensors)...")
from transformers import AutoConfig, AutoModelForCausalLM
hf_cfg = AutoConfig.from_pretrained(MODEL_DIR)
# Force eager attention so we can compare bit-close (SDPA / flash may differ).
t0 = time.time()
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,       # dequant everything to fp32 for a clean comparison
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
hf_model.eval()
print(f"  loaded in {time.time()-t0:.1f}s")

# Sanity: check the layer's weights are what we expect
hf_layer = hf_model.model.layers[LAYER_IDX]
print(f"\nHF layer input_layernorm.weight[:4] = "
      f"{hf_layer.input_layernorm.weight[:4].tolist()}")
print(f"Our lw.input_layernorm_weight[:4]     = "
      f"{lw.input_layernorm_weight[:4].tolist()}")

# ============ Build the test input ============
torch.manual_seed(0xDECAFBAD)
B, S, D = 1, 8, cfg.hidden_size
x = (torch.randn(B, S, D) * 0.1).float()

# Precompute RoPE cos/sin for positions [0, S).
positions = torch.arange(S, dtype=torch.long)
cos, sin = compute_cos_sin_for_positions(hf_cfg, positions)
print(f"\nRoPE cos: {list(cos.shape)}  sin: {list(sin.shape)}")

# ============ Run OUR forward pass ============
print("\nRunning our Vulkan-backed layer forward...")
t0 = time.time()
with torch.no_grad():
    y_ours = gpt_oss_layer_forward(ext, x, LAYER_IDX, lw, cfg, cos, sin)
print(f"  elapsed: {time.time()-t0:.2f}s")
print(f"  y_ours: {list(y_ours.shape)}  mean={y_ours.mean():.4f}  std={y_ours.std():.4f}  "
      f"range=[{y_ours.min():.3f}, {y_ours.max():.3f}]")

# ============ Run REFERENCE forward pass ============
print("\nRunning transformers reference for layer 0 only...")
# The layer expects (hidden_states, attention_mask, position_ids, ...)
# We need to feed it position_ids so RoPE gets computed properly inside.
position_ids = torch.arange(S).unsqueeze(0)

# Build a full causal mask (for full attention layers).
# The layer_types[0] setting decides which mask HF picks up; let's give it the
# right one manually and pass in a position_embeddings tuple to bypass HF's
# lazy computation.
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
rot = GptOssRotaryEmbedding(hf_cfg)
hf_cos, hf_sin = rot(torch.zeros(1, S, D), position_ids)   # HF's own cos/sin

# Causal mask [B, 1, S, S], 0 for kept, -inf for masked
mask = torch.zeros(B, 1, S, S)
mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))

t0 = time.time()
with torch.no_grad():
    y_ref, *_ = hf_layer(
        hidden_states=x,
        attention_mask=mask,
        position_ids=position_ids,
        position_embeddings=(hf_cos, hf_sin),
    )
print(f"  elapsed: {time.time()-t0:.2f}s")
print(f"  y_ref:  {list(y_ref.shape)}  mean={y_ref.mean():.4f}  std={y_ref.std():.4f}  "
      f"range=[{y_ref.min():.3f}, {y_ref.max():.3f}]")

# ============ Compare ============
diff = (y_ours - y_ref).abs()
max_abs = diff.max().item()
rel     = diff / y_ref.abs().clamp_min(1e-6)
max_rel = rel.max().item()

print(f"\n============================")
print(f"max|diff|      = {max_abs:.4e}")
print(f"max|diff/ref|  = {max_rel:.4e}")
print(f"mean|diff|     = {diff.mean().item():.4e}")

# Bit-exact impossible due to FP32 accum reordering. FP32 layer forward across
# ~25M+ operations should be ~1e-4..1e-3.
ok = max_abs < 5e-3
print(f"\n{'PASS' if ok else 'FAIL'}   (threshold 5e-3 abs)")
