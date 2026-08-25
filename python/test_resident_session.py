"""Validate incremental resident session state and exact token-ID appends."""
from __future__ import annotations
import os,pathlib,sys,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');torch.set_num_threads(12)
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import GptOssModel
from gpt_oss.resident_session import ResidentDecodeSession
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b');sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
model=GptOssModel.from_pretrained(ext,MODEL,stream_experts=True);model.enable_streamed_vram_cache(18,'lfu');model.pin_lm_head_to_vram();model.pin_projections_to_vram(False);tok=AutoTokenizer.from_pretrained(MODEL);prompt_ids=tok.encode('The capital of France is')
with ResidentDecodeSession(model,tok,64) as a:
 a.prefill_token_ids(prompt_ids);one=a.generate(4)
with ResidentDecodeSession(model,tok,64) as b:
 b.prefill_token_ids(prompt_ids);split=b.generate(2)+b.generate(2);print('split_equal',split==one,one,split);assert split==one
 append_ids=tok.encode(' Continue:')
 predicted=b.append_token_ids(append_ids)
 full_ids=b.token_ids
 logits=model.forward(torch.tensor([full_ids]),only_last_logits=True);ref=int(logits[0,-1].argmax());print('append_prediction',predicted,ref,predicted==ref,'position',b.position,len(full_ids));assert predicted==ref;assert b.position==len(full_ids)
 continuation=b.generate(2);print('continuation',continuation,'decode',repr(b.decode()));assert continuation[0]==ref
print('RESIDENT_SESSION_OK')
