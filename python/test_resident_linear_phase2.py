"""Validate resident RMSNorm -> linear projection on real 20b weights."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');torch.set_num_threads(12)
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import ResidentTensor
from gpt_oss.layer import rmsnorm
from gpt_oss.loader import Safetensors,load_layer
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-20b');sdk=pathlib.Path(os.environ['VULKAN_SDK'])
ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
lw=load_layer(Safetensors(MODEL),0,load_experts=False);torch.manual_seed(22);x=torch.randn(1,2880)*.05;norm=rmsnorm(x,lw.input_layernorm_weight,1e-5);ref=norm@lw.q_proj_weight.T+lw.q_proj_bias;N=lw.q_proj_weight.shape[0];owned=[]
def own(t):r=ResidentTensor.from_tensor(ext,t);owned.append(r);return r
def empty(shape):r=ResidentTensor.empty(ext,shape);owned.append(r);return r
rx=own(x);rw=own(lw.input_layernorm_weight);rn=empty(x.shape);qw=own(lw.q_proj_weight);qb=own(lw.q_proj_bias);qy=empty((1,N));ext.rmsnorm_resident(rx.handle,rw.handle,rn.handle,1,2880,1e-5);ext.linear_resident_io(rn.handle,qw.handle,qb.handle,qy.handle,1,N,2880,True);got=qy.download();d=(got-ref).abs();print('max',d.max().item(),'mean',d.mean().item(),'ok',torch.allclose(got,ref,rtol=1e-4,atol=1e-4));assert torch.allclose(got,ref,rtol=1e-4,atol=1e-4)
for _ in range(3):ext.rmsnorm_resident(rx.handle,rw.handle,rn.handle,1,2880,1e-5);ext.linear_resident_io(rn.handle,qw.handle,qb.handle,qy.handle,1,N,2880,True)
n=20;t=time.perf_counter()
for _ in range(n):ext.rmsnorm_resident(rx.handle,rw.handle,rn.handle,1,2880,1e-5);ext.linear_resident_io(rn.handle,qw.handle,qb.handle,qy.handle,1,N,2880,True)
print('resident_ms',(time.perf_counter()-t)*1000/n);t=time.perf_counter()
for _ in range(n):rmsnorm(x,lw.input_layernorm_weight,1e-5)@lw.q_proj_weight.T+lw.q_proj_bias
print('cpu_ms',(time.perf_counter()-t)*1000/n)
for r in owned:r.free()
print('RESIDENT_LINEAR_PHASE2_OK')
