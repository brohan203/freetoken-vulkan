"""Validate resident [S,H,D] <-> [H,S,D] transpose."""
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE));from gpt_oss import ResidentTensor
sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False);torch.manual_seed(91)
for S,H,D in [(1,64,64),(5,64,64),(5,8,64),(32,64,64)]:
 x=torch.randn(S,H,D);ref=x.permute(1,0,2).contiguous()
 with ResidentTensor.from_tensor(ext,x) as a,ResidentTensor.empty(ext,ref.shape) as b,ResidentTensor.empty(ext,x.shape) as c:
  ext.transpose_resident(a.handle,b.handle,S,H,D,False);got=b.download();ext.transpose_resident(b.handle,c.handle,S,H,D,True);back=c.download()
 print(S,H,D,torch.equal(got,ref),torch.equal(back,x));assert torch.equal(got,ref);assert torch.equal(back,x)
print('RESIDENT_TRANSPOSE_OK')
