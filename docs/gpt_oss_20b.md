# gpt-oss-20b port to Vulkan - architecture and plan

## Current status

**Session 1:**
- [DONE] Downloaded gpt-oss-20b MXFP4 weights (12.84 GB, 3 safetensors chunks)
- [DONE] Inspected tensor layout - 32 experts, GQA 64:8, D=2880, moe_intermediate=2880
- [DONE] Wrote MXFP4 dequant reference in PyTorch - **bit-exact match** with `transformers.integrations.mxfp4._convert_moe_packed_tensors`
- [DONE] Journal entries: `e028` (MXFP4 format), `e029` (gpt-oss-20b arch)

**Session 2 (this session):**
- [DONE] Enabled 8-bit storage + int8 in Vulkan context
- [DONE] Wrote `shaders/mxfp4_matvec_f32.comp` - MXFP4 weight x- FP32 activation matvec
- [DONE] Extension binding `mxfp4_matvec` - bit-close vs PyTorch on real gpt-oss expert weights (max abs err 1.5e-6)
- [DONE] Decoded gpt-oss's exact activation from `transformers.GptOssExperts._apply_gate` - interleaved gate/up, asymmetric clamps, `(up+1) * gate * sigmoid(gate*1.702)`, biases (e030)
- [DONE] Wrote `shaders/moe_mlp_mxfp4_gpt_oss_f32.comp` - full gpt-oss MoE MLP with MXFP4
- [DONE] Extension binding `moe_mlp_gpt_oss` - **runs on real gpt-oss-20b layer-0 weights, matches transformers reference to 2.86e-06** (e031)

**Session 3 (this session):**
- [DONE] Extended flash attention with GQA + causal + sliding window + attention sinks (`flash_attention_gpt_oss_f32.comp`), verified 8/8 at real gpt-oss shapes (max err 3e-8, e032)
- [DONE] Partial RoPE (`rope_partial_f32.comp`), verified 5/5 (e033)
- [DONE] Extension bindings `flash_attention_gpt_oss` and `rope_partial`

**All shader kernels needed for gpt-oss-20b inference are DONE.**

Next: integration - HuggingFace loader, tokenizer, decode loop.

## Model architecture (verified from config.json + safetensors)

```
Layers: 24, alternating [sliding_attn, sliding_attn, full_attn, ...]
  - Full attention layers: 8  (every 3rd starting from layer 2)
  - Sliding window layers: 16 (window=128)

Hidden D:            2880
Head dim:            64
Q heads:             64  (Q  proj: [4096, 2880])
KV heads:            8   (KV proj: [512, 2880])
GQA ratio:           8:1

MoE:
  Experts:           32
  Top-K:             4
  Per-expert Dff:    2880 (per gate/per up)
  Fused gate_up:     5760 out per expert (concatenated gate+up)
  MoE weights:       MXFP4 (4.25 bpw)
  Router:            [32, 2880] BF16

Attention:
  Q/K/V/O weights:   BF16 (NOT quantized)
  Attention sinks:   [64] BF16 per layer, learnable per-head softmax bias
  RoPE:              YARN-scaled (original 4096, extended to 131072)
  Partial rotary:    factor 0.25 (only 25% of head_dim rotated)

Vocab:               201088 (o200k_harmony tokenizer)
Context length:      131072
Activation:          SwiGLU with alpha=1.70 clamp
Norm:                RMSNorm (pre-norm)
```

## Component checklist (dependencies)

### Kernels needed (some done, some TODO)

| Kernel | Status | Notes |
|---|---|---|
| RMSNorm | [DONE] have | Existing shader |
| BF16 GEMM (attention proj) | [TODO] skip for now, dequant on load | ~1.9 GB extra memory, acceptable |
| MXFP4 dequant -> FP32 | [DONE] done | `mxfp4_matvec_f32.comp`, verified vs HF |
| MoE MLP with MXFP4 experts + activation + biases | [DONE] done | `moe_mlp_mxfp4_gpt_oss_f32.comp`, matches real gpt-oss layer-0 weights (max err 2.86e-6) |
| MoE router (E=32, top-4) | [DONE] have | Extend our existing router - same interface |
| Flash attention MH + GQA + causal + SWA + sinks | [DONE] done | `flash_attention_gpt_oss_f32.comp`, matches gpt-oss config (max err 3e-8) |
| Partial YARN RoPE | [DONE] done | `rope_partial_f32.comp`, HF-compatible rotate_half (max err 5e-7) |
| Embedding lookup | [TODO] CPU gather for now | Trivial to do CPU-side |

### Non-kernel infrastructure

| Piece | Status |
|---|---|
| MXFP4 dequant reference (PyTorch, bit-exact) | [DONE] done - `python/gpt_oss/mxfp4_ref.py` |
| Safetensors loader (in-process, streaming) | [TODO] TODO |
| o200k_harmony tokenizer (via `tiktoken`) | [TODO] TODO - install `tiktoken` package |
| KV cache management | [TODO] TODO |
| Sampling loop | [TODO] TODO |
| End-to-end forward pass | [TODO] TODO |

## Realistic timeline

Per honest scope from earlier discussion: ~4-6 weeks of focused work for
gpt-oss-20b working end-to-end. Session 1 hit ~1 day of that (weights
downloaded, format understood, PyTorch reference done). Next sessions
build the shader work.

## Fully resident decode status

The generic `python/chat_gpt_oss.py` supports a fully resident 20b path:

- all 24 layers of FP32 Q/K/V/O/router and norm weights are resident;
- all 32 experts per layer fit in the resident expert cache;
- CPU prefill transfers KV once, then single-token decode stays resident;
- 64-token output exactly matches the legacy path;
- canonical global expert IDs feed the pinned 32-expert tables directly, with
  no CPU cache policy or ID remapping;
- 64-token runtime improved from `40.91 s` legacy to `3.31 s` resident after
  additionally fusing the full pre-MoE projection/attention/router segment;
- resident decode averaged `0.0340 s/token` (about 29.4 tokens/second);
- a 320-token stress averaged `0.0371 s/token` with resident bytes unchanged at
  `15,053,650,176` and no OOM.

## Historical order of operations

1. **[NEXT] MXFP4 dequant helper in GLSL** - matches Python reference bit-for-bit
2. **MXFP4 -> FP16 matmul shader** - replaces our reg_tiled_f16 for expert weights
3. **MoE MLP with MXFP4** - extend moe_mlp_lg to read MXFP4 experts
4. **GQA + sinks in attention** - extend flash_attention_mh
5. **YARN RoPE + sliding window** - new small shaders
6. **Persistent weight upload** - put everything in VRAM once at init
7. **HuggingFace tokenizer + text I/O** - via tiktoken
8. **KV cache + decode loop** - Python glue
9. **End-to-end test** - generate a sentence
