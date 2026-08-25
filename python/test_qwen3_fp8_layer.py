"""Validate one dequantized Qwen3-8B-FP8 layer."""
from __future__ import annotations
import os,pathlib,sys,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from qwen3 import ShardedSafetensors,compute_rope,load_qwen3_config,load_qwen3_layer,qwen3_layer_forward
from test_qwen3_layer import reference_layer
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\Qwen3-8B-FP8')
def main():
 torch.set_num_threads(12);sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False);cfg=load_qwen3_config(MODEL);w=load_qwen3_layer(ShardedSafetensors(MODEL),0);torch.manual_seed(151);x=torch.randn(1,3,cfg.hidden_size)*.02;cos,sin=compute_rope(cfg,torch.arange(3));ref=reference_layer(x,w,cfg,cos,sin);got=qwen3_layer_forward(ext,x,w,cfg,cos,sin);d=(got-ref).abs();print('shape',tuple(got.shape),'max',d.max().item(),'mean',d.mean().item(),'finite',torch.isfinite(got).all().item());assert torch.allclose(got,ref,rtol=1e-4,atol=5e-4);print('QWEN3_FP8_LAYER_OK')
if __name__=='__main__':main()
