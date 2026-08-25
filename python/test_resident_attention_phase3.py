"""Validate resident RoPE, KV append, and capacity-strided attention."""
from __future__ import annotations
import os,pathlib,sys,math,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import ResidentTensor
from gpt_oss.layer import _cpu_rope,_cpu_decode_attention
sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
torch.manual_seed(31);B,Hq,Hkv,D,capacity=1,64,8,64,16;scale=1/math.sqrt(D);sinks=torch.randn(Hq)*.2;owned=[]
def own(t):r=ResidentTensor.from_tensor(ext,t);owned.append(r);return r
def empty(shape):r=ResidentTensor.empty(ext,shape);owned.append(r);return r
kc=empty((B,Hkv,capacity,D));vc=empty((B,Hkv,capacity,D));rsinks=own(sinks);cpu_k=[];cpu_v=[]
for position in range(5):
 q=torch.randn(B,Hq,1,D)*.1;k=torch.randn(B,Hkv,1,D)*.1;v=torch.randn(B,Hkv,1,D)*.1;freq=torch.randn(1,D);cos=freq.cos();sin=freq.sin();rq=own(q);rk=own(k);rv=own(v);rc=own(cos);rs=own(sin);qr=empty(q.shape);kr=empty(k.shape);out=empty(q.shape)
 ext.rope_kv_attention_resident(rq.handle,rk.handle,rv.handle,rc.handle,rs.handle,qr.handle,kr.handle,kc.handle,vc.handle,rsinks.handle,out.handle,B,Hq,Hkv,1,D,position,capacity,128,True,scale)
 qcpu=_cpu_rope(q,cos,sin);kcpu=_cpu_rope(k,cos,sin);cpu_k.append(kcpu);cpu_v.append(v);ref=_cpu_decode_attention(qcpu,torch.cat(cpu_k,2),torch.cat(cpu_v,2),sinks,scale,128);got=out.download();d=(got-ref).abs();print(position,d.max().item(),d.mean().item(),torch.allclose(got,ref,rtol=1e-3,atol=1e-4));assert torch.allclose(got,ref,rtol=1e-3,atol=1e-4)
 for r in [rq,rk,rv,rc,rs,qr,kr,out]:r.free()
# Cache layout prefix verification by downloading full slab.
kg=kc.download();vg=vc.download();assert torch.allclose(kg[:,:,:5,:],torch.cat(cpu_k,2),rtol=1e-5,atol=1e-6);assert torch.equal(vg[:,:,:5,:],torch.cat(cpu_v,2));print('cache_ok')
for r in owned:r.free()
print('RESIDENT_ATTENTION_PHASE3_OK')
