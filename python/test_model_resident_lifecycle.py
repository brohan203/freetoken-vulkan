"""Verify model-owned resident allocations are released exactly once."""
from __future__ import annotations
import os,pathlib,sys,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');torch.set_num_threads(12)
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import GptOssModel,greedy_generate_resident
M=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-20b');sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False);base=ext.resident_bytes_total();model=GptOssModel.from_pretrained(ext,M,stream_experts=True);model.enable_streamed_vram_cache(32,'lfu');model.pin_lm_head_to_vram();model.pin_projections_to_vram(False);tok=AutoTokenizer.from_pretrained(M);_,ids,_=greedy_generate_resident(model,tok,'Hello',2,32,False);allocated=ext.resident_bytes_total();print('base',base,'allocated',allocated,'ids',ids);assert allocated>base;model.close();print('after_close',ext.resident_bytes_total());assert ext.resident_bytes_total()==base;model.close();assert ext.resident_bytes_total()==base
try:model.forward(torch.tensor([[1]]));raise AssertionError('closed model should reject forward')
except RuntimeError:print('closed_guard_ok')
print('MODEL_RESIDENT_LIFECYCLE_OK')
