"""Run a full lazy Qwen3-8B-FP8 next-token smoke."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from qwen3 import Qwen3Model
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\Qwen3-8B-FP8')
def main():
 torch.set_num_threads(12);sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False);tok=AutoTokenizer.from_pretrained(MODEL);model=Qwen3Model.from_pretrained(ext,MODEL);ids=tok.encode('The capital of France is',return_tensors='pt').long();t=time.time();logits=model.forward(ids,only_last_logits=True,collect_layer_times=True);elapsed=time.time()-t;vals,idx=torch.topk(logits[0,-1],10);top=[(int(i),float(v),tok.decode([int(i)])) for v,i in zip(vals,idx)];print('shape',tuple(logits.shape),'finite',torch.isfinite(logits).all().item(),'elapsed',elapsed,'layer_avg',sum(model.layer_times)/len(model.layer_times),'top',top);assert torch.isfinite(logits).all();assert tok.decode([int(idx[0])]).strip()=='Paris';print('QWEN3_FP8_FULL_MODEL_OK')
if __name__=='__main__':main()
