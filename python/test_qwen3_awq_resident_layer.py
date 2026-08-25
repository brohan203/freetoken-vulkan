"""Compare resident AWQ Qwen3-14B layer with dequantized reference."""
from __future__ import annotations
import os,pathlib,sys,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from dense_kv_cache import DenseKVCache
from qwen3 import ShardedSafetensors,compute_rope,load_qwen3_config,load_qwen3_layer,qwen3_layer_forward
from qwen3.resident import ResidentQwen3Weights,ResidentQwen3Workspace,resident_qwen3_layer
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\Qwen3-14B-AWQ')
def main():
 torch.set_num_threads(12);sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False);base=ext.resident_bytes_total();cfg=load_qwen3_config(MODEL);sf=ShardedSafetensors(MODEL);w=load_qwen3_layer(sf,0);torch.manual_seed(181);x=torch.randn(1,1,cfg.hidden_size)*.02;cos,sin=compute_rope(cfg,torch.tensor([0]));cache=DenseKVCache(cfg,max_seqlen=16);ref=qwen3_layer_forward(ext,x,w,cfg,cos,sin,layer_idx=0,past_kv=cache,past_len=0);rw=ResidentQwen3Weights(ext);layer=rw.append_awq(sf,0);ws=ResidentQwen3Workspace(ext,cfg,16);slot=ws.upload_input(x,cos,sin);slot=resident_qwen3_layer(ext,ws,layer,0,slot,0);got=ws.hidden[slot].download().reshape_as(ref);d=(got-ref).abs();print('max',d.max().item(),'mean',d.mean().item(),'resident',ext.resident_bytes_total());assert torch.allclose(got,ref,rtol=1e-4,atol=5e-4);ws.free();rw.free();assert ext.resident_bytes_total()==base;print('QWEN3_AWQ_RESIDENT_LAYER_OK')
if __name__=='__main__':main()
