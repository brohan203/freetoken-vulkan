"""Validate fully resident gpt-oss-20b parity, speed, and stability."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');torch.set_num_threads(12)
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import GptOssModel,greedy_generate_resident
from gpt_oss.generate import greedy_generate_kv
M=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-20b');sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
m=GptOssModel.from_pretrained(ext,M,stream_experts=True);m.enable_streamed_vram_cache(32,'lfu');m.pin_lm_head_to_vram();m.pin_projections_to_vram(False);tok=AutoTokenizer.from_pretrained(M);prompt='The capital of France is';t=time.time();_,legacy,_=greedy_generate_kv(m,tok,prompt,64,384,False);legacy_s=time.time()-t;t=time.time();_,resident,stats=greedy_generate_resident(m,tok,prompt,64,384,False);resident_s=time.time()-t;print('parity',legacy==resident,'legacy',legacy_s,'resident',resident_s,'decode',sum(stats['decode_times'])/len(stats['decode_times']));assert legacy==resident
before=ext.resident_bytes_total();t=time.time();text,ids,long=greedy_generate_resident(m,tok,'Continue this list: 1.',320,384,False);elapsed=time.time()-t;after=ext.resident_bytes_total();print('stress',len(ids),elapsed,sum(long['decode_times'])/len(long['decode_times']),'bytes',before,after,'text_len',len(text));assert len(ids)==320;assert before==after;print('RESIDENT_20B_GENERATION_OK')
