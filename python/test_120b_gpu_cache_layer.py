"""Verify one 120b layer: CPU compact streaming versus GPU expert cache."""
from __future__ import annotations
import os,pathlib,sys,time
os.environ.setdefault("VULKAN_SDK",r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"]=os.path.join(os.environ["VULKAN_SDK"],"Bin")+os.pathsep+os.environ.get("PATH","")
os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import torch
from torch.utils.cpp_extension import load
from transformers import AutoConfig
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss.config import GptOssConfig
from gpt_oss.layer import gpt_oss_layer_forward
from gpt_oss.loader import Safetensors,ExpertStore,load_layer
from gpt_oss.rope import compute_cos_sin_for_positions
from gpt_oss.streaming_resident import StreamedResidentMoECache
MODEL=pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-120b");sdk=pathlib.Path(os.environ["VULKAN_SDK"])
ext=load(name="freetoken_vulkan_ext",sources=[str(HERE/"ext_module.cpp")],extra_include_paths=[str(sdk/"Include"),str(REPO/"include")],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}","vulkan-1.lib"],extra_cflags=["/O2","/D_CRT_SECURE_NO_WARNINGS"],verbose=False)
cfg=GptOssConfig.from_json(MODEL/'config.json');hf=AutoConfig.from_pretrained(MODEL);sf=Safetensors(MODEL);layer=load_layer(sf,0,load_experts=False)
torch.manual_seed(120);x=torch.randn(1,1,cfg.hidden_size)*.01;cos,sin=compute_cos_sin_for_positions(hf,torch.tensor([0]))
store_cpu=ExpertStore(sf,cache_size_per_layer=16);t0=time.time();a=gpt_oss_layer_forward(ext,x,0,layer,cfg,cos,sin,expert_store=store_cpu);cpu_s=time.time()-t0
store_gpu=ExpertStore(sf,cache_size_per_layer=24);gpu=StreamedResidentMoECache(ext,1,slots_per_layer=24);t0=time.time();b=gpt_oss_layer_forward(ext,x,0,layer,cfg,cos,sin,expert_store=store_gpu,streamed_resident=gpu);cold_s=time.time()-t0
t0=time.time();c=gpt_oss_layer_forward(ext,x,0,layer,cfg,cos,sin,expert_store=store_gpu,streamed_resident=gpu);warm_s=time.time()-t0
print(f"cpu_s={cpu_s:.6f} gpu_cold_s={cold_s:.6f} gpu_warm_s={warm_s:.6f}")
for name,y in [('cold',b),('warm',c)]:
 d=(a-y).abs();ok=torch.allclose(a,y,rtol=1e-5,atol=5e-5);print(name,'max',d.max().item(),'mean',d.mean().item(),'ok',ok);assert ok
print(f"gpu_hits={gpu.hits} gpu_misses={gpu.misses} upload_gib={gpu.uploaded_bytes/1024**3:.4f} upload_s={gpu.upload_seconds:.4f}")
print('GPU_CACHE_LAYER_WITHIN_FP32_TOLERANCE')
