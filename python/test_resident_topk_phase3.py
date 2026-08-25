"""Validate resident 128-way top-4 softmax router operation."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import ResidentTensor
sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
torch.manual_seed(33);logits=torch.randn(7,128);logits[0,3]=logits[0,7]=5.0;vals,ids=torch.topk(logits,4,-1);weights=torch.softmax(vals,-1)
with ResidentTensor.from_tensor(ext,logits) as rl,ResidentTensor.empty(ext,(7,4)) as rw:
 ih=ext.allocate_resident(7*4*4);ext.topk_resident(rl.handle,ih,rw.handle,7);got_i=ext.download_resident_i32(ih,[7,4]).long();got_w=rw.download();ext.free_resident(ih)
print('ids_equal',torch.equal(got_i,ids),got_i[0].tolist(),ids[0].tolist());d=(got_w-weights).abs();print('weights',d.max().item(),d.mean().item(),torch.allclose(got_w,weights,rtol=1e-5,atol=1e-6));assert torch.equal(got_i,ids);assert torch.allclose(got_w,weights,rtol=1e-5,atol=1e-6);print('RESIDENT_TOPK_PHASE3_OK')
