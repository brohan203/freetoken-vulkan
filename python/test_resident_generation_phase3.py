"""Compare 64 resident decode tokens and stress 320 tokens without OOM."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');torch.set_num_threads(12)
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import GptOssModel,greedy_generate_resident
from gpt_oss.generate import greedy_generate_kv
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b');sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
model=GptOssModel.from_pretrained(ext,MODEL,stream_experts=True);model.enable_streamed_vram_cache(18,'lfu');model.pin_lm_head_to_vram();model.pin_projections_to_vram(False);tok=AutoTokenizer.from_pretrained(MODEL);prompt='The capital of France is'
t=time.time();_,legacy,_=greedy_generate_kv(model,tok,prompt,max_new_tokens=64,max_seqlen=384,print_stream=False);legacy_s=time.time()-t
t=time.time();text,resident,stats=greedy_generate_resident(model,tok,prompt,64,384,False);resident_s=time.time()-t;print('compare64',legacy==resident,'legacy_s',legacy_s,'resident_s',resident_s,'decode',sum(stats['decode_times'])/len(stats['decode_times']));assert legacy==resident
# Long resident stress uses same model/cache and verifies bounded resident bytes.
before=ext.resident_bytes_total();t=time.time();long_text,long_ids,long_stats=greedy_generate_resident(model,tok,'Continue this numbered list: 1.',320,384,False);elapsed=time.time()-t;after=ext.resident_bytes_total();print('stress320_tokens',len(long_ids),'elapsed',elapsed,'decode',sum(long_stats['decode_times'])/len(long_stats['decode_times']),'resident_before',before,'resident_after',after,'finite_text_len',len(long_text));assert len(long_ids)==320;assert before==after;print('RESIDENT_GENERATION_PHASE3_OK')
