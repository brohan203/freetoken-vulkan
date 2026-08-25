"""Compare Qwen3 KV-cached greedy generation with Transformers."""
from __future__ import annotations
import gc,os,pathlib,sys,time
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import torch
from torch.utils.cpp_extension import load
from transformers import AutoModelForCausalLM,AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from qwen3 import Qwen3Model,greedy_generate
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\Qwen3-4B');REPORT=HERE/'qwen3_generation_report.txt'
def main():
 torch.set_num_threads(12);tok=AutoTokenizer.from_pretrained(MODEL);prompt='The capital of France is';inputs=tok.encode(prompt,return_tensors='pt').long();t=time.time();ref_model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,low_cpu_mem_usage=True);ref=ref_model.generate(inputs,max_new_tokens=8,do_sample=False,use_cache=True);ref_s=time.time()-t;ref_ids=ref[0,inputs.shape[1]:].tolist();del ref_model;gc.collect();sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False);model=Qwen3Model.from_pretrained(ext,MODEL);t=time.time();text,ids,stats=greedy_generate(model,tok,prompt,8,64,False);ours_s=time.time()-t;lines=[f'reference_ids={ref_ids!r}',f'actual_ids={ids!r}',f'equal={ref_ids==ids}',f'reference_text={tok.decode(ref[0].tolist())!r}',f'actual_text={text!r}',f'reference_seconds={ref_s:.6f}',f'actual_seconds={ours_s:.6f}',f'prefill_seconds={stats["prefill_seconds"]:.6f}',f'decode_average={sum(stats["decode_times"])/len(stats["decode_times"]):.6f}',f'cache_bytes={stats["cache_bytes"]}'];REPORT.write_text('\n'.join(lines),encoding='ascii',errors='backslashreplace');assert ref_ids==ids;print('QWEN3_GENERATION_OK')
if __name__=='__main__':main()
