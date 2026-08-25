"""Validate batched resident prefill for one real 120b layer."""
from __future__ import annotations
import os,pathlib,sys,time,torch,math
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');torch.set_num_threads(12)
from torch.utils.cpp_extension import load
from transformers import AutoConfig
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import ResidentTensor
from gpt_oss.config import GptOssConfig
from gpt_oss.layer import gpt_oss_layer_forward
from gpt_oss.loader import Safetensors,ExpertStore,load_layer,ModelWeights
from gpt_oss.resident_projections import ResidentProjectionWeights
from gpt_oss.rope import compute_cos_sin_for_positions
from gpt_oss.streaming_resident import StreamedResidentMoECache
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b');sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
cfg=GptOssConfig.from_json(MODEL/'config.json');hf=AutoConfig.from_pretrained(MODEL);sf=Safetensors(MODEL);lw=load_layer(sf,0,load_experts=False);store=ExpertStore(sf);S=5;torch.manual_seed(92);x=torch.randn(1,S,2880)*.02;cos,sin=compute_cos_sin_for_positions(hf,torch.arange(S));ref_cache=StreamedResidentMoECache(ext,1,32,'lfu');ref=gpt_oss_layer_forward(ext,x,0,lw,cfg,cos,sin,expert_store=store,streamed_resident=ref_cache)
modelw=ModelWeights(cfg,torch.empty(0),torch.empty(0),torch.empty(0),layers=[lw],expert_store=store);projection_manager=ResidentProjectionWeights(ext,modelw,False);proj=projection_manager.for_layer(0);own=[]
def fromt(t):r=ResidentTensor.from_tensor(ext,t);own.append(r);return r
def empty(shape):r=ResidentTensor.empty(ext,shape);own.append(r);return r
rx=fromt(x.reshape(S,2880));norm=empty((S,2880));qf=empty((S,4096));kf=empty((S,512));vf=empty((S,512));qh=empty((64,S,64));kh=empty((8,S,64));vh=empty((8,S,64));rc=fromt(cos);rs=fromt(sin);qr=empty((64,S,64));kr=empty((8,S,64));kc=empty((8,S,64));vc=empty((8,S,64));attn=empty((64,S,64));attnf=empty((S,4096));projected=empty((S,2880));residual=empty((S,2880));post=empty((S,2880));logits=empty((S,128));weights=empty((S,4));indices=ext.allocate_resident(S*4*4);hidden=empty((S,4,2880));moe=empty((S,2880));out=empty((S,2880));cache=StreamedResidentMoECache(ext,1,32,'lfu')
t=time.perf_counter();ext.rmsnorm_qkv_resident(rx.handle,proj.input_norm,norm.handle,proj.q_weight,proj.q_bias,qf.handle,proj.k_weight,proj.k_bias,kf.handle,proj.v_weight,proj.v_bias,vf.handle,S,2880,4096,512,1e-5);ext.transpose_resident(qf.handle,qh.handle,S,64,64,False);ext.transpose_resident(kf.handle,kh.handle,S,8,64,False);ext.transpose_resident(vf.handle,vh.handle,S,8,64,False);ext.rope_kv_attention_resident(qh.handle,kh.handle,vh.handle,rc.handle,rs.handle,qr.handle,kr.handle,kc.handle,vc.handle,proj.sinks,attn.handle,1,64,8,S,64,0,S,128,True,1/math.sqrt(64));ext.transpose_resident(attn.handle,attnf.handle,S,64,64,True);ext.oproj_router_resident(attnf.handle,proj.o_weight,proj.o_bias,projected.handle,rx.handle,residual.handle,proj.post_norm,post.handle,proj.router_weight,proj.router_bias,logits.handle,S,2880,4096,128,1e-5);ext.topk_resident(logits.handle,indices,weights.handle,S);global_ids=ext.download_resident_i32(indices,[S,4]).long();cache.call_resident(0,store,post.handle,global_ids,weights.handle,indices,hidden.handle,moe.handle,S);ext.add_resident(residual.handle,moe.handle,out.handle,S*2880);elapsed=time.perf_counter()-t;got=out.download().reshape_as(ref);d=(got-ref).abs();print('elapsed',elapsed,'max',d.max().item(),'mean',d.mean().item(),'ok',torch.allclose(got,ref,rtol=1e-4,atol=5e-4),'unique',torch.unique(global_ids).numel());assert torch.allclose(got,ref,rtol=1e-4,atol=5e-4)
for r in own:
 r.free()
ext.free_resident(indices)
cache.free()
projection_manager.free()
print('RESIDENT_PREFILL_LAYER_OK')
