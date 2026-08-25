"""Validate fused resident O projection, residual, norm, and router."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');torch.set_num_threads(12)
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import ResidentTensor
from gpt_oss.layer import rmsnorm
from gpt_oss.loader import Safetensors,load_layer
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b');sdk=pathlib.Path(os.environ['VULKAN_SDK'])
ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
lw=load_layer(Safetensors(MODEL),0,load_experts=False);torch.manual_seed(24);attn=torch.randn(1,lw.o_proj_weight.shape[1])*.02;residual=torch.randn(1,2880)*.02;proj=attn@lw.o_proj_weight.T+lw.o_proj_bias;res=proj+residual;norm=rmsnorm(res,lw.post_attention_layernorm_weight,1e-5);logits=norm@lw.router_weight.T+lw.router_bias;ref_top=torch.topk(logits,4,-1).indices;owned=[]
def own(t):r=ResidentTensor.from_tensor(ext,t);owned.append(r);return r
def empty(shape):r=ResidentTensor.empty(ext,shape);owned.append(r);return r
ra=own(attn);ow=own(lw.o_proj_weight);ob=own(lw.o_proj_bias);rp=empty(proj.shape);rr=own(residual);rres=empty(res.shape);nw=own(lw.post_attention_layernorm_weight);rn=empty(norm.shape);routerw=own(lw.router_weight);routerb=own(lw.router_bias);rl=empty(logits.shape);args=[ra.handle,ow.handle,ob.handle,rp.handle,rr.handle,rres.handle,nw.handle,rn.handle,routerw.handle,routerb.handle,rl.handle,1,2880,lw.o_proj_weight.shape[1],128,1e-5];ext.oproj_router_resident(*args);got=[rp.download(),rres.download(),rn.download(),rl.download()]
for name,a,b,tol in zip(['proj','res','norm','logits'],got,[proj,res,norm,logits],[1e-4,1e-4,1e-4,2e-4]):d=(a-b).abs();print(name,d.max().item(),d.mean().item(),torch.allclose(a,b,rtol=1e-4,atol=tol));assert torch.allclose(a,b,rtol=1e-4,atol=tol)
print('top4',torch.topk(got[-1],4,-1).indices.tolist(),ref_top.tolist());assert torch.equal(torch.topk(got[-1],4,-1).indices,ref_top)
for _ in range(3):ext.oproj_router_resident(*args)
n=20;t=time.perf_counter()
for _ in range(n):ext.oproj_router_resident(*args)
print('fused_ms',(time.perf_counter()-t)*1000/n);t=time.perf_counter()
for _ in range(n):
 pp=attn@lw.o_proj_weight.T+lw.o_proj_bias;rs=pp+residual;nn=rmsnorm(rs,lw.post_attention_layernorm_weight,1e-5);ll=nn@lw.router_weight.T+lw.router_bias
print('cpu_ms',(time.perf_counter()-t)*1000/n)
for r in owned:r.free()
print('RESIDENT_OPROJ_ROUTER_PHASE2_OK')
