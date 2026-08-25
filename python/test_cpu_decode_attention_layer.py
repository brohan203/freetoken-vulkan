"""Compare CPU and Vulkan decode-time attention in a real 120b layer."""
from __future__ import annotations
import os,pathlib,sys,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');torch.set_num_threads(12)
from torch.utils.cpp_extension import load
from transformers import AutoConfig
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss.config import GptOssConfig
from gpt_oss.kv_cache import KVCache
from gpt_oss.loader import Safetensors,ExpertStore,load_layer
from gpt_oss.layer import gpt_oss_layer_forward,_cpu_rope,_cpu_decode_attention
from gpt_oss.rope import compute_cos_sin_for_positions
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b');sdk=pathlib.Path(os.environ['VULKAN_SDK'])
ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
cfg=GptOssConfig.from_json(MODEL/'config.json');hf=AutoConfig.from_pretrained(MODEL);sf=Safetensors(MODEL);lw=load_layer(sf,0,load_experts=False);store=ExpertStore(sf);torch.manual_seed(55);past=7;B=1;Hq=64;Hkv=8;D=64;scale=D**-.5;q=torch.randn(B,Hq,1,D)*.1;k=torch.randn(B,Hkv,1,D)*.1;v=torch.randn(B,Hkv,1,D)*.1;pk=torch.randn(B,Hkv,past,D)*.1;pv=torch.randn(B,Hkv,past,D)*.1;cos,sin=compute_cos_sin_for_positions(hf,torch.tensor([past]));qcpu=_cpu_rope(q,cos,sin);kcpu=_cpu_rope(k,cos,sin);K=torch.cat([pk,kcpu],dim=2);V=torch.cat([pv,v],dim=2);a=_cpu_decode_attention(qcpu,K,V,lw.sinks,scale,cfg.sliding_window);qv=ext.rope_partial(q,cos,sin,D);kv=ext.rope_partial(k,cos,sin,D);b=ext.flash_attention_gpt_oss_kv(qv,torch.cat([pk,kv],dim=2),V,lw.sinks,scale,past,cfg.sliding_window,True);d=(a-b).abs();print('max',d.max().item(),'mean',d.mean().item(),'ok',torch.allclose(a,b,rtol=1e-3,atol=1e-4));assert torch.allclose(a,b,rtol=1e-3,atol=1e-4);print('CPU_DECODE_ATTENTION_OK')
