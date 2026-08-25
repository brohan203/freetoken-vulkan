"""Direct GPU equivalence test for full versus compact 120b expert tables."""
from __future__ import annotations
import os, pathlib, sys, time
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ.get("PATH", "")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent
REPO=HERE.parent
sys.path.insert(0,str(HERE))
from gpt_oss.loader import Safetensors, ExpertStore, load_layer
MODEL=pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-120b")
sdk=pathlib.Path(os.environ["VULKAN_SDK"])
ext=load(name="freetoken_vulkan_ext",sources=[str(HERE/"ext_module.cpp")],extra_include_paths=[str(sdk/"Include"),str(REPO/"include")],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}","vulkan-1.lib"],extra_cflags=["/O2","/D_CRT_SECURE_NO_WARNINGS"],verbose=False)
sf=Safetensors(MODEL)
eager=load_layer(sf,0,load_experts=True)
store=ExpertStore(sf)
torch.manual_seed(7)
x=torch.randn(1,2880,dtype=torch.float32)*0.01
idx=torch.tensor([[98,82,102,99]],dtype=torch.int64)
w=torch.softmax(torch.tensor([[3.2,3.1,2.9,2.8]],dtype=torch.float32),dim=-1)
compact,local=store.materialize_selected(0,idx)
print('global',idx.tolist(),'local',local.tolist(),flush=True)
t0=time.time(); y_full=ext.moe_mlp_gpt_oss(x,idx,w,eager.gate_up_blocks,eager.gate_up_scales,eager.gate_up_bias,eager.down_blocks,eager.down_scales,eager.down_bias); print('full_s',time.time()-t0,flush=True)
t0=time.time(); y_compact=ext.moe_mlp_gpt_oss(x,local,w,compact.gate_up_blocks,compact.gate_up_scales,compact.gate_up_bias,compact.down_blocks,compact.down_scales,compact.down_bias); print('compact_s',time.time()-t0,flush=True)
d=(y_full-y_compact).abs(); print('max',d.max().item(),'mean',d.mean().item(),'equal',torch.equal(y_full,y_compact),flush=True)
# Also prove one-expert calls independently.
for slot in range(4):
 gi=idx[:,slot:slot+1].contiguous(); li=local[:,slot:slot+1].contiguous(); ww=torch.ones(1,1)
 yf=ext.moe_mlp_gpt_oss(x,gi,ww,eager.gate_up_blocks,eager.gate_up_scales,eager.gate_up_bias,eager.down_blocks,eager.down_scales,eager.down_bias)
 yc=ext.moe_mlp_gpt_oss(x,li,ww,compact.gate_up_blocks,compact.gate_up_scales,compact.gate_up_bias,compact.down_blocks,compact.down_scales,compact.down_bias)
 dd=(yf-yc).abs(); print('slot',slot,'global',gi.item(),'local',li.item(),'max',dd.max().item(),'equal',torch.equal(yf,yc),flush=True)
