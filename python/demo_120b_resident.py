"""Optimized gpt-oss-120b resident decode demo."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
torch.set_num_threads(int(os.environ.get('FREETOKEN_CPU_THREADS','12')))
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import GptOssModel,greedy_generate_resident
MODEL=pathlib.Path(os.environ.get('FREETOKEN_MODEL_DIR',r'C:\Users\rohanborkar\Downloads\gpt-oss-120b'));PROMPT=os.environ.get('FREETOKEN_PROMPT','The capital of France is');N=int(os.environ.get('FREETOKEN_MAX_NEW','48'));sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
t=time.time();model=GptOssModel.from_pretrained(ext,MODEL,stream_experts=True);model.enable_streamed_vram_cache(int(os.environ.get('FREETOKEN_CACHE_SLOTS','18')),'lfu');model.pin_lm_head_to_vram(os.environ.get('FREETOKEN_FP16_LM_HEAD','0')=='1');model.pin_projections_to_vram(False);print('setup_s',time.time()-t,'resident_gib',ext.resident_bytes_total()/1024**3);tok=AutoTokenizer.from_pretrained(MODEL);t=time.time();text,ids,stats=greedy_generate_resident(model,tok,PROMPT,N,256,False);total=time.time()-t;print('text',repr(text));print('tokens',ids);print('prefill',stats['prefill_time'],'decode_avg',sum(stats['decode_times'])/max(1,len(stats['decode_times'])),'total',total,'resident_gib',ext.resident_bytes_total()/1024**3)
