"""Compare fully resident Qwen3 model step with verified lazy model."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from qwen3 import Qwen3Model,ResidentQwen3Workspace,resident_qwen3_model_step
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\Qwen3-4B')
def main():
 torch.set_num_threads(12);sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False);base=ext.resident_bytes_total();model=Qwen3Model.from_pretrained(ext,MODEL);tok=AutoTokenizer.from_pretrained(MODEL);token=tok.encode('Hello')[0];ref=model.forward(torch.tensor([[token]]),only_last_logits=True);t=time.time();model.pin_to_vram(False);pin=time.time()-t;ws=ResidentQwen3Workspace(ext,model.config,64);t=time.time();got=resident_qwen3_model_step(model,ws,token,0);step=time.time()-t;d=(got-ref[0]).abs();ref_top=torch.topk(ref[0,-1],10).indices.tolist();got_top=torch.topk(got[0],10).indices.tolist();print('pin',pin,'step',step,'resident',ext.resident_bytes_total(),'max',d.max().item(),'mean',d.mean().item(),'top_equal',ref_top==got_top,'top',got_top);assert ref_top==got_top;ws.free();model.close();print('after',ext.resident_bytes_total());assert ext.resident_bytes_total()==base;print('QWEN3_RESIDENT_MODEL_OK')
if __name__=='__main__':main()
