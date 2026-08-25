"""Validate fused resident RMSNorm + QKV on real 20b weights."""
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
lw=load_layer(Safetensors(MODEL),0,load_experts=False);torch.manual_seed(23);x=torch.randn(1,2880)*.05;n=rmsnorm(x,lw.input_layernorm_weight,1e-5);refs=[n@w.T+b for w,b in [(lw.q_proj_weight,lw.q_proj_bias),(lw.k_proj_weight,lw.k_proj_bias),(lw.v_proj_weight,lw.v_proj_bias)]];owned=[]
def own(t):r=ResidentTensor.from_tensor(ext,t);owned.append(r);return r
def empty(shape):r=ResidentTensor.empty(ext,shape);owned.append(r);return r
rx=own(x);nw=own(lw.input_layernorm_weight);rn=empty(x.shape);weights=[own(t) for t in [lw.q_proj_weight,lw.q_proj_bias,lw.k_proj_weight,lw.k_proj_bias,lw.v_proj_weight,lw.v_proj_bias]];q=empty(refs[0].shape);k=empty(refs[1].shape);v=empty(refs[2].shape);args=[rx.handle,nw.handle,rn.handle,weights[0].handle,weights[1].handle,q.handle,weights[2].handle,weights[3].handle,k.handle,weights[4].handle,weights[5].handle,v.handle,1,2880,refs[0].shape[1],refs[1].shape[1],1e-5];ext.rmsnorm_qkv_resident(*args);got=[q.download(),k.download(),v.download()]
for name,a,b in zip('qkv',got,refs):d=(a-b).abs();print(name,d.max().item(),d.mean().item(),torch.allclose(a,b,rtol=1e-4,atol=1e-4));assert torch.allclose(a,b,rtol=1e-4,atol=1e-4)
for _ in range(3):ext.rmsnorm_qkv_resident(*args)
niter=20;t=time.perf_counter()
for _ in range(niter):ext.rmsnorm_qkv_resident(*args)
print('fused_ms',(time.perf_counter()-t)*1000/niter);t=time.perf_counter()
for _ in range(niter):
 nn=rmsnorm(x,lw.input_layernorm_weight,1e-5);aa=nn@lw.q_proj_weight.T+lw.q_proj_bias;bb=nn@lw.k_proj_weight.T+lw.k_proj_bias;cc=nn@lw.v_proj_weight.T+lw.v_proj_bias
print('cpu_ms',(time.perf_counter()-t)*1000/niter)
for r in owned:r.free()
print('RESIDENT_QKV_PHASE2_OK')
