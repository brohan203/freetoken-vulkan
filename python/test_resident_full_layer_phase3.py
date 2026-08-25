"""Validate complete resident single-token 120b layer."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');torch.set_num_threads(12)
from torch.utils.cpp_extension import load
from transformers import AutoConfig
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss.loader import load_model
from gpt_oss.layer import gpt_oss_layer_forward
from gpt_oss.rope import compute_cos_sin_for_positions
from gpt_oss.resident_projections import ResidentProjectionWeights
from gpt_oss.streaming_resident import StreamedResidentMoECache
from gpt_oss.resident_decode import ResidentDecodeWorkspace,resident_decode_layer
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b');sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
w=load_model(MODEL,layers=[0],stream_experts=True);cfg=w.config;hf=AutoConfig.from_pretrained(MODEL);torch.manual_seed(40);x=torch.randn(1,1,cfg.hidden_size)*.02;cos,sin=compute_cos_sin_for_positions(hf,torch.tensor([0]));cpu_cache=StreamedResidentMoECache(ext,1,18,'lfu');ref=gpt_oss_layer_forward(ext,x,0,w.layers[0],cfg,cos,sin,expert_store=w.expert_store,streamed_resident=cpu_cache)
projections=ResidentProjectionWeights(ext,w,False);gpu_cache=StreamedResidentMoECache(ext,1,18,'lfu');ws=ResidentDecodeWorkspace(ext,cfg,1,16);current=ws.upload_input(x);ws.upload_rope(cos,sin);t=time.perf_counter();current,ids=resident_decode_layer(ext,ws,0,current,projections.for_layer(0),w.expert_store,gpu_cache,0,cfg.sliding_window);cold=time.perf_counter()-t;got=ws.hidden[current].download().reshape_as(ref);d=(got-ref).abs();print('cold',cold,'max',d.max().item(),'mean',d.mean().item(),'ok',torch.allclose(got,ref,rtol=1e-4,atol=5e-4),'ids',ids.tolist());assert torch.allclose(got,ref,rtol=1e-4,atol=5e-4)
# Re-upload same input and run warm cache repeatedly.
ws.upload_input(x);current=0
for _ in range(2):current,_=resident_decode_layer(ext,ws,0,current,projections.for_layer(0),w.expert_store,gpu_cache,0,cfg.sliding_window);ws.upload_input(x);current=0
n=20;t=time.perf_counter()
for _ in range(n):current,_=resident_decode_layer(ext,ws,0,current,projections.for_layer(0),w.expert_store,gpu_cache,0,cfg.sliding_window);ws.upload_input(x);current=0
resident_ms=(time.perf_counter()-t)*1000/n;t=time.perf_counter()
for _ in range(n):gpt_oss_layer_forward(ext,x,0,w.layers[0],cfg,cos,sin,expert_store=w.expert_store,streamed_resident=cpu_cache)
current_ms=(time.perf_counter()-t)*1000/n;print('resident_ms',resident_ms,'current_ms',current_ms,'speedup',current_ms/resident_ms);ws.free();projections.free();gpu_cache.free();cpu_cache.free();print('RESIDENT_FULL_LAYER_PHASE3_OK')
