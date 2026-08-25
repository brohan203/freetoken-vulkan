"""Validate resident RMSNorm using a real gpt-oss-20b norm weight."""
from __future__ import annotations
import os,pathlib,sys,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import ResidentTensor
from gpt_oss.layer import rmsnorm
from gpt_oss.loader import Safetensors,load_layer
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-20b');sdk=pathlib.Path(os.environ['VULKAN_SDK'])
ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
layer=load_layer(Safetensors(MODEL),0,load_experts=False);torch.manual_seed(20);x=torch.randn(1,2880)*.05;ref=rmsnorm(x,layer.input_layernorm_weight,1e-5)
with ResidentTensor.from_tensor(ext,x) as rx,ResidentTensor.from_tensor(ext,layer.input_layernorm_weight) as rw,ResidentTensor.empty(ext,x.shape) as out:
 ext.rmsnorm_resident(rx.handle,rw.handle,out.handle,1,2880,1e-5);got=out.download()
d=(got-ref).abs();print('max',d.max().item(),'mean',d.mean().item(),'ok',torch.allclose(got,ref,rtol=1e-5,atol=1e-5));assert torch.allclose(got,ref,rtol=1e-5,atol=1e-5);print('RESIDENT_20B_ACTIVATION_OK')
