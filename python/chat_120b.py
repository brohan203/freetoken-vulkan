"""Long-lived gpt-oss-120b prompt loop with persistent expert-cache warmth."""
from __future__ import annotations
import argparse,os,pathlib,sys,time
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0')
os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import torch
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import GptOssModel
from gpt_oss.generate import greedy_generate_kv

def build_model(model_dir:pathlib.Path,slots:int,threads:int):
 torch.set_num_threads(threads);sdk=pathlib.Path(os.environ['VULKAN_SDK'])
 ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
 model=GptOssModel.from_pretrained(ext,model_dir,stream_experts=True);model.enable_streamed_vram_cache(slots,'lfu');model.pin_lm_head_to_vram();return model,AutoTokenizer.from_pretrained(model_dir)

def run_prompt(model,tokenizer,prompt,max_new,max_seq):
 h0=model.streamed_resident.hits;m0=model.streamed_resident.misses;t0=time.perf_counter();text,ids,stats=greedy_generate_kv(model,tokenizer,prompt,max_new_tokens=max_new,max_seqlen=max_seq,print_stream=True);elapsed=time.perf_counter()-t0;h=model.streamed_resident.hits-h0;m=model.streamed_resident.misses-m0;print(f'[stats] total={elapsed:.2f}s prefill={stats["prefill_time"]:.2f}s decode={sum(stats["decode_times"])/max(1,len(stats["decode_times"])):.3f}s/token cache={100*h/max(1,h+m):.1f}%');return text

def main():
 parser=argparse.ArgumentParser();parser.add_argument('prompt',nargs='?');parser.add_argument('--model-dir',type=pathlib.Path,default=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b'));parser.add_argument('--max-new-tokens',type=int,default=48);parser.add_argument('--max-seq-len',type=int,default=256);parser.add_argument('--slots',type=int,default=24);parser.add_argument('--threads',type=int,default=12);args=parser.parse_args();model,tokenizer=build_model(args.model_dir,args.slots,args.threads)
 if args.prompt is not None:run_prompt(model,tokenizer,args.prompt,args.max_new_tokens,args.max_seq_len);return
 print('gpt-oss-120b ready. Empty line exits.')
 while True:
  try:prompt=input('\nprompt> ').strip()
  except (EOFError,KeyboardInterrupt):break
  if not prompt:break
  run_prompt(model,tokenizer,prompt,args.max_new_tokens,args.max_seq_len)
if __name__=='__main__':main()
