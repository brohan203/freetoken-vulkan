"""Compare resident Qwen3-8B-FP8 generation with lazy dequantized path."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from qwen3 import Qwen3Model,ResidentQwen3Workspace,greedy_generate,greedy_generate_resident
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\Qwen3-8B-FP8')
def main():
 torch.set_num_threads(12);sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False);base=ext.resident_bytes_total();tok=AutoTokenizer.from_pretrained(MODEL);model=Qwen3Model.from_pretrained(ext,MODEL);prompt='The capital of France is';t=time.time();ref_text,ref_ids,_=greedy_generate(model,tok,prompt,8,64,False);lazy=time.time()-t;t=time.time();model.pin_to_vram(False);pin=time.time()-t;ws=ResidentQwen3Workspace(ext,model.config,64);steady=ext.resident_bytes_total();t=time.time();text,ids,stats=greedy_generate_resident(model,tok,prompt,8,64,False,ws);elapsed=time.time()-t;after=ext.resident_bytes_total();avg=sum(stats['decode_times'])/len(stats['decode_times']);print('ref',ref_ids,repr(ref_text),'ids',ids,'equal',ids==ref_ids,'text',repr(text),'lazy',lazy,'pin',pin,'total',elapsed,'prompt_avg',sum(stats['prompt_times'])/len(stats['prompt_times']),'decode_avg',avg,'bytes',steady,after);assert ids==ref_ids;assert steady==after;ws.free();model.close();assert ext.resident_bytes_total()==base;print('QWEN3_FP8_RESIDENT_GENERATION_OK')
if __name__=='__main__':main()
