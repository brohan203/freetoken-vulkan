"""Phase-1 resident activation, RMSNorm, and residual-add validation."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import ResidentTensor
from gpt_oss.layer import rmsnorm
sdk=pathlib.Path(os.environ['VULKAN_SDK'])
ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
torch.manual_seed(71);rows,hidden=4,2880;x=torch.randn(rows,hidden)*.02;weight=torch.randn(hidden)*.1+1;residual=torch.randn(rows,hidden)*.02;base=ext.resident_bytes_total();rx=ResidentTensor.from_tensor(ext,x);rw=ResidentTensor.from_tensor(ext,weight);rr=ResidentTensor.from_tensor(ext,residual);rn=ResidentTensor.empty(ext,x.shape);ro=ResidentTensor.empty(ext,x.shape)
ext.rmsnorm_resident(rx.handle,rw.handle,rn.handle,rows,hidden,1e-5);got_norm=rn.download();ref_norm=rmsnorm(x,weight,1e-5);d=(got_norm-ref_norm).abs();print('rmsnorm',d.max().item(),d.mean().item(),torch.allclose(got_norm,ref_norm,rtol=1e-5,atol=1e-5));assert torch.allclose(got_norm,ref_norm,rtol=1e-5,atol=1e-5)
ext.add_resident(rn.handle,rr.handle,ro.handle,x.numel());got=ro.download();gpu_add_ref=got_norm+residual;cpu_chain_ref=ref_norm+residual;d=(got-gpu_add_ref).abs();end=(got-cpu_chain_ref).abs();print('add_exact',d.max().item(),torch.equal(got,gpu_add_ref),'chain_max',end.max().item());assert torch.equal(got,gpu_add_ref);assert torch.allclose(got,cpu_chain_ref,rtol=1e-5,atol=1e-5)
# Reuse and timing.
for _ in range(5):ext.rmsnorm_resident(rx.handle,rw.handle,rn.handle,rows,hidden,1e-5);ext.add_resident(rn.handle,rr.handle,ro.handle,x.numel())
n=20;t=time.perf_counter()
for _ in range(n):ext.rmsnorm_resident(rx.handle,rw.handle,rn.handle,rows,hidden,1e-5);ext.add_resident(rn.handle,rr.handle,ro.handle,x.numel())
print('resident_chain_ms',(time.perf_counter()-t)*1000/n)
t=time.perf_counter()
for _ in range(n):rmsnorm(x,weight,1e-5)+residual
print('cpu_chain_ms',(time.perf_counter()-t)*1000/n)
allocated=ext.resident_bytes_total()-base;expected=(x.numel()*4*4+weight.numel()*4);print('allocated',allocated,'expected',expected);assert allocated==expected
for tensor in [rx,rw,rr,rn,ro]:tensor.free()
print('after_free',ext.resident_bytes_total()-base);assert ext.resident_bytes_total()==base
try:rx.download();raise AssertionError('freed tensor download should fail')
except RuntimeError:print('freed_guard_ok')
print('RESIDENT_PHASE1_OK')
