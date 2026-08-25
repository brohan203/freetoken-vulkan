"""Verify Qwen3 layer KV prefill + decode equals a full causal pass."""
from __future__ import annotations
import os,pathlib,sys,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from dense_kv_cache import DenseKVCache
from qwen3 import ShardedSafetensors,compute_rope,load_qwen3_config,load_qwen3_layer,qwen3_layer_forward
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\Qwen3-4B')
def main():
 torch.set_num_threads(12);sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False);cfg=load_qwen3_config(MODEL);w=load_qwen3_layer(ShardedSafetensors(MODEL),0);torch.manual_seed(111);x=torch.randn(1,4,cfg.hidden_size)*.02;cos,sin=compute_rope(cfg,torch.arange(4));full=qwen3_layer_forward(ext,x,w,cfg,cos,sin);cache=DenseKVCache(cfg,max_seqlen=16);c0,s0=compute_rope(cfg,torch.arange(3));prefill=qwen3_layer_forward(ext,x[:,:3],w,cfg,c0,s0,layer_idx=0,past_kv=cache,past_len=0);cache.advance(3);c1,s1=compute_rope(cfg,torch.tensor([3]));decode=qwen3_layer_forward(ext,x[:,3:],w,cfg,c1,s1,layer_idx=0,past_kv=cache,past_len=3);d=(decode-full[:,3:]).abs();print('prefill_shape',tuple(prefill.shape),'decode_shape',tuple(decode.shape),'max',d.max().item(),'mean',d.mean().item(),'cache_len',cache.cur_len);assert torch.allclose(decode,full[:,3:],rtol=1e-4,atol=5e-4);print('QWEN3_KV_LAYER_OK')
if __name__=='__main__':main()
