"""Verify all projection weights fit with LM head and 18-slot expert cache."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');torch.set_num_threads(12)
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import GptOssModel
from gpt_oss.generate import greedy_generate_kv
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b');sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
model=GptOssModel.from_pretrained(ext,MODEL,stream_experts=True);model.pin_lm_head_to_vram();before=ext.resident_bytes_total();model.pin_projections_to_vram();after=ext.resident_bytes_total();print('projection_gib',(after-before)/1024**3,'base_gib',before/1024**3,'total_gib',after/1024**3);assert 3.5<(after-before)/1024**3<3.8;model.enable_streamed_vram_cache(18,'lfu');tok=AutoTokenizer.from_pretrained(MODEL);text,ids,stats=greedy_generate_kv(model,tok,'Hello',max_new_tokens=2,max_seqlen=32,print_stream=False);final=ext.resident_bytes_total()/1024**3;print('final_gib',final,'tokens',ids,'text',repr(text));assert final<15.5;print('RESIDENT_PROJECTION_BUDGET_OK')
