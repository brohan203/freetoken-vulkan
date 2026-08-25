"""Verify pin_all_to_vram rejects streamed-expert checkpoints safely."""
import os,pathlib,sys
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','')
from torch.utils.cpp_extension import load
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE));from gpt_oss import GptOssModel
M=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b');sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False);base=ext.resident_bytes_total();model=GptOssModel.from_pretrained(ext,M,stream_experts=True)
try:model.pin_all_to_vram();raise AssertionError('streamed model pin_all must fail')
except RuntimeError as e:print('rejected',str(e))
model.close();assert ext.resident_bytes_total()==base;print('PIN_ALL_CONTRACT_OK')
