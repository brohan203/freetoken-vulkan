"""Validate resident activation/control I/O for two-stage MoE."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import ResidentTensor
from gpt_oss.loader import Safetensors,ExpertStore
from gpt_oss.resident import ResidentLayerHandles
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b');sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
store=ExpertStore(Safetensors(MODEL));global_ids=[3,47,91,126];mapped=store.mapped_experts(0,global_ids);experts=[mapped[i] for i in global_ids];cat=lambda n:torch.cat([getattr(e,n) for e in experts],0).contiguous();weights=[cat(n) for n in ['gate_up_blocks','gate_up_scales','gate_up_bias','down_blocks','down_scales','down_bias']];handles=[ext.upload_resident(t) for t in weights];rh=ResidentLayerHandles(*handles,E=4,D=2880,Dff=2880);torch.manual_seed(35);x=torch.randn(1,2880)*.02;ids=torch.tensor([[0,1,2,3]],dtype=torch.int32);routing=torch.softmax(torch.randn(1,4),-1);ref=rh.call(ext,x,ids.long(),routing,two_stage=True)
rx=ResidentTensor.from_tensor(ext,x);rid=ext.upload_resident(ids);rw=ResidentTensor.from_tensor(ext,routing);hidden=ResidentTensor.empty(ext,(1,4,2880));out=ResidentTensor.empty(ext,(1,2880));args=[rx.handle,rid,rw.handle,hidden.handle,out.handle,*handles,4,2880,2880,1,4];ext.moe_mlp_gpt_oss_twostage_io(*args);got=out.download();d=(got-ref).abs();print('max',d.max().item(),'mean',d.mean().item(),'ok',torch.equal(got,ref));assert torch.equal(got,ref)
for _ in range(3):ext.moe_mlp_gpt_oss_twostage_io(*args)
n=20;t=time.perf_counter()
for _ in range(n):ext.moe_mlp_gpt_oss_twostage_io(*args)
print('resident_io_ms',(time.perf_counter()-t)*1000/n);t=time.perf_counter()
for _ in range(n):rh.call(ext,x,ids.long(),routing,two_stage=True)
print('transient_io_ms',(time.perf_counter()-t)*1000/n)
for h in handles+[rid]:ext.free_resident(h)
for r in [rx,rw,hidden,out]:r.free()
print('RESIDENT_MOE_IO_PHASE3_OK')
