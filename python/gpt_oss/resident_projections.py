"""Resident FP32 projection and normalization weights for decode layers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .loader import ModelWeights,LayerWeights

@dataclass
class ResidentProjectionLayer:
 input_norm:int;q_weight:int;q_bias:int;k_weight:int;k_bias:int;v_weight:int;v_bias:int;o_weight:int;o_bias:int;post_norm:int;router_weight:int;router_bias:int;sinks:int
 def handles(self):return [self.input_norm,self.q_weight,self.q_bias,self.k_weight,self.k_bias,self.v_weight,self.v_bias,self.o_weight,self.o_bias,self.post_norm,self.router_weight,self.router_bias,self.sinks]

class ResidentProjectionWeights:
 def __init__(self,ext,weights:ModelWeights,verbose:bool=True):
  import time
  self.ext=ext;self.layers:List[ResidentProjectionLayer]=[];t=time.time()
  for i,layer in enumerate(weights.layers):
   self.layers.append(self._upload(layer))
   if verbose and (i+1)%6==0:print(f'  [projections {i+1}/{len(weights.layers)}] {ext.resident_bytes_total()/1024**3:.2f} GiB total resident')
  if verbose:print(f'[Resident projections] {len(self.layers)} layers in {time.time()-t:.1f}s')
 def _upload(self,l:LayerWeights):
  upload=self.ext.upload_resident
  return ResidentProjectionLayer(*(upload(t) for t in [l.input_layernorm_weight,l.q_proj_weight,l.q_proj_bias,l.k_proj_weight,l.k_proj_bias,l.v_proj_weight,l.v_proj_bias,l.o_proj_weight,l.o_proj_bias,l.post_attention_layernorm_weight,l.router_weight,l.router_bias,l.sinks]))
 def for_layer(self,i:int):return self.layers[i]
 def free(self):
  for layer in self.layers:
   for h in layer.handles():self.ext.free_resident(h)
  self.layers.clear()
 def __del__(self):
  try:self.free()
  except Exception:pass
