"""Compare complete 120b LM-head FP16 and FP32 resident matvecs."""
from __future__ import annotations
import os,pathlib,sys,torch,time
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import ResidentTensor
from gpt_oss.loader import Safetensors
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b');sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
w=Safetensors(MODEL).get('lm_head.weight');torch.manual_seed(81);x=torch.randn(1,2880)*.02;ref=x@w.float().T
with ResidentTensor.from_tensor(ext,x) as rx,ResidentTensor.empty(ext,(1,w.shape[0])) as y16,ResidentTensor.empty(ext,(1,w.shape[0])) as y32:
 h16=ext.upload_resident(w.half().contiguous());h32=ext.upload_resident(w.float().contiguous());ext.linear_fp16_resident_io(rx.handle,h16,0,y16.handle,1,w.shape[0],w.shape[1],False);ext.linear_resident_io(rx.handle,h32,0,y32.handle,1,w.shape[0],w.shape[1],False);a=y16.download();b=y32.download();ext.free_resident(h16);ext.free_resident(h32)
for name,y in [('fp16',a),('fp32',b)]:
 d=(y-ref).abs();print(name,'max',d.max().item(),'mean',d.mean().item(),'top',int(y.argmax()),'ref_top',int(ref.argmax()))
cross=(a-b).abs();ok=torch.allclose(a,b,rtol=1e-4,atol=1e-4) and int(a.argmax())==int(b.argmax());print('cross',cross.max().item(),ok);assert ok;print('FP16_LM_HEAD_OK')
