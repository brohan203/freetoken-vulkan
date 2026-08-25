"""Compare resident and existing gpt-oss-120b decode token sequences."""
from __future__ import annotations
import os,pathlib,sys,time,torch
os.environ.setdefault('VULKAN_SDK',r'C:\VulkanSDK\1.4.357.0');os.environ['PATH']=os.path.join(os.environ['VULKAN_SDK'],'Bin')+os.pathsep+os.environ.get('PATH','');torch.set_num_threads(12)
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parent;sys.path.insert(0,str(HERE))
from gpt_oss import GptOssModel
from gpt_oss.generate import greedy_generate_kv
from gpt_oss.resident_decode import ResidentDecodeWorkspace,resident_decode_model_step
MODEL=pathlib.Path(r'C:\Users\rohanborkar\Downloads\gpt-oss-120b');sdk=pathlib.Path(os.environ['VULKAN_SDK']);ext=load(name='freetoken_vulkan_ext',sources=[str(HERE/'ext_module.cpp')],extra_include_paths=[str(sdk/'Include'),str(REPO/'include')],extra_ldflags=[f"/LIBPATH:{sdk/'Lib'}",'vulkan-1.lib'],extra_cflags=['/O2','/D_CRT_SECURE_NO_WARNINGS'],verbose=False)
model=GptOssModel.from_pretrained(ext,MODEL,stream_experts=True);model.enable_streamed_vram_cache(18,'lfu');model.pin_lm_head_to_vram();model.pin_projections_to_vram(False);tok=AutoTokenizer.from_pretrained(MODEL);prompt='The capital of France is';_,ref_ids,_=greedy_generate_kv(model,tok,prompt,max_new_tokens=4,max_seqlen=32,print_stream=False);input_ids=tok.encode(prompt,return_tensors='pt').long();cache=model.make_kv_cache(32);t=time.perf_counter();prefill=model.forward(input_ids,past_kv=cache,past_len=0,only_last_logits=True);cache.advance(input_ids.shape[1]);first=int(prefill[0,-1].argmax());ws=ResidentDecodeWorkspace(ext,model.cfg,model.cfg.num_hidden_layers,32);ws.load_kv_cache(cache);ids=[first];position=cache.cur_len;step_times=[]
for _ in range(3):
 t0=time.perf_counter();logits,_=resident_decode_model_step(model,ws,ids[-1],position);step_times.append(time.perf_counter()-t0);ids.append(int(logits[0].argmax()));position+=1
elapsed=time.perf_counter()-t;print('ref',ref_ids,'resident',ids,'equal',ref_ids==ids,'step_times',step_times,'text',repr(tok.decode(tok.encode(prompt)+ids)));assert ref_ids==ids;print('resident_gib',ext.resident_bytes_total()/1024**3);ws.free();print('RESIDENT_FULL_MODEL_PHASE3_OK')
