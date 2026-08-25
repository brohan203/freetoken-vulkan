# gpt-oss-120b on 16 GB AMD GPUs

Status: working end-to-end with streamed MXFP4 experts.

## Verified configuration

- Model: `openai/gpt-oss-120b` root MXFP4 checkpoint
- GPU target: AMD Radeon RX 6800 XT, 16 GB VRAM
- Host RAM: 64 GB
- Architecture: 36 layers, 128 experts per layer, top-4 routing
- Hidden size, attention geometry, intermediate size, RoPE, attention sinks,
  and activation are identical to gpt-oss-20b. Existing compute shaders are
  reused unchanged.

## Why eager loading does not fit

The MXFP4 checkpoint is approximately 60.8 GiB. Eagerly converting attention,
embedding, router, and LM-head BF16 tensors to FP32 while keeping every expert
in CPU memory exceeds a 64 GiB host budget.

The 120b path therefore uses:

1. file-backed safetensors expert slices;
2. router-first expert selection;
3. global-to-local expert ID remapping;
4. a bounded per-layer device-local expert cache;
5. the existing KV cache for decode.

## Expert streaming

`ExpertStore` keeps MXFP4 expert tensors file-backed and materializes only the
unique experts selected by the router. Single-token decode selects at most four
experts per layer, approximately 50.5 MiB, instead of the full approximately
1.58 GiB layer expert table.

Correctness was verified on real 120b layer-0 weights:

- eager 128-expert table versus compact selected table: bit-exact;
- full layer eager versus compact streamed layer: bit-exact;
- CPU compact versus device-resident expert cache: max absolute difference
  approximately `3.05e-5`, mean approximately `7e-7`, within the established
  FP32 accumulation tolerance.

## Device-local expert cache

`StreamedResidentMoECache` allocates fixed expert slots per layer. The default
is 24 slots per layer:

- approximately 10.6 GiB maximum expert-slab VRAM;
- each cache miss reads one expert from safetensors and uploads it into a slot;
- each hit remaps the global router ID directly to the existing slot;
- misses for one layer invocation are uploaded in one Vulkan submission;
- least-recently-used experts are evicted while protecting experts selected by
  the current invocation.

Long prefill can select more unique experts than the slot count. In that case,
the layer falls back to one compact transient expert table for the invocation.
Chunked prefill was tested and found slower because repeated layer forwards and
launches outweighed cache reuse; one-shot prefill remains the default.

## Measured behavior

Prompt: `The capital of France is`

Generated continuation:

```text
 Paris.

Great! If you have any more questions or need further assistance,
```

Representative results:

| Configuration | Prompt/new tokens | Prefill | Decode average | Total |
|---|---:|---:|---:|---:|
| CPU compact streaming | 5 / 6 | 15.4 s | 4.1 s/token | 41.2 s |
| 16-entry CPU LRU | 5 / 6 | 10.8 s | 3.0 s/token | 30.0 s |
| 24-slot GPU expert cache | 5 / 6 | 9.9 s | 3.0 s/token | 30.1 s |
| CPU compact streaming | 5 / 16 | 14.2 s | 3.7 s/token | 69.7 s |
| 24-slot GPU expert cache | 5 / 16 | 9.2 s | 3.1 s/token | 48.2 s |
| 24-slot GPU cache + mmap expert rows | 5 / 16 | 7.8 s | 2.3 s/token | 41.8 s |
| 24-slot GPU cache + mmap rows + resident LM head | 5 / 16 | 6.9 s | 2.09 s/token | 38.2 s |
| Previous row + 12 PyTorch CPU threads | 5 / 16 | 6.5 s | 1.65 s/token | 31.2 s |
| Previous row + two-stage parallel MoE | 5 / 16 | 6.7 s | 0.54 s/token | 15.4 s |
| Two-stage parallel MoE, 48 tokens | 5 / 48 | 5.9 s | 0.506 s/token | 29.7 s |
| Previous row + CPU decode attention | 5 / 48 | 6.2 s | 0.486 s/token | 29.1 s |
| Previous row + parallel staging memcpy | 5 / 48 | 4.3 s | 0.394 s/token | 22.8 s |
| Fully resident decode, 64 tokens | 5 / 64 | CPU prefill | 0.229 s/token | 15.8 s |
| Fully resident 320-token stress | 7 / 320 | CPU prefill | 0.272 s/token | 95.7 s |
| 28-slot GPU cache + mmap expert rows | 5 / 16 | 7.7 s | 2.3 s/token | 41.8 s |

Twenty-four slots are the default because 28 slots consume more VRAM without a
measurable gain. GPU-cache misses now upload contiguous memory-mapped expert
rows directly into pooled Vulkan staging buffers. Only the small BF16 bias rows
are converted to FP32. Host-visible upload staging buffers are reused through
the Vulkan buffer pool. Together these changes reduce measured host expert
materialization from approximately 18 seconds to approximately 0.2 seconds in
the 16-token run. Resident LM head is now enabled by
default because the faster expert path makes its approximately 7 percent total
speedup measurable while fitting within the 16 GB budget. Twelve PyTorch
intra-op CPU threads are the measured default for the remaining Q/K/V/O/router
matrix-vector work; 6, 8, and 12 threads were tested end-to-end and 12 produced
the best total (31.2 seconds for 16 generated tokens).

The default eviction policy is LFU with recency tie-breaking. A 48-token trace
measured 64.8 percent LFU hits versus 61.7 percent for LRU under the same
24-slot-per-layer VRAM budget. End-to-end time fell from 113.5 seconds with LRU
to 101.4 seconds with LFU, with identical output tokens. LRU remains available
through `FREETOKEN_CACHE_POLICY=lru`.

## Running

Edit paths if needed, then run:

```powershell
python\demo_120b_generate.py
```

Optional environment variables:

```text
FREETOKEN_PROMPT
FREETOKEN_MAX_NEW
FREETOKEN_CACHE_SLOTS
FREETOKEN_CACHE_POLICY
FREETOKEN_GPU_CACHE
FREETOKEN_PIN_LM_HEAD
FREETOKEN_PREFILL_CHUNK
FREETOKEN_CPU_THREADS
```

For repeated prompts, `python/chat_120b.py` keeps the model and expert cache
alive across requests. In a measured four-request sequence, the first
`The capital of France is` request took about 10.0 seconds; repeating the same
prompt later in the process took about 6.2 seconds. Prefill fell from 3.66 to
0.99 seconds and cache hit rate rose from 53.5 to 66.8 percent.

Recommended defaults:

```text
FREETOKEN_CACHE_SLOTS=18
FREETOKEN_CACHE_POLICY=lfu
FREETOKEN_GPU_CACHE=1
FREETOKEN_PIN_LM_HEAD=1
FREETOKEN_PREFILL_CHUNK=0
FREETOKEN_CPU_THREADS=12
```

## Fully resident decode

The default `chat_120b.py` path keeps single-token decode resident across all
36 layers:

1. resident RMSNorm plus Q/K/V projections;
2. resident RoPE, capacity-strided KV append, and attention;
3. resident O projection, residual, post-attention norm, and router;
4. resident top-4 weights, with only 16 bytes of global IDs downloaded for
   expert-cache remapping;
5. resident-input/output two-stage MoE and final residual;
6. resident final norm and LM head.

All FP32 projection/norm/router/sink weights consume 3.61 GiB. Together with
the FP32 LM head and an 18-slot expert cache, final measured resident usage is
13.77 GiB. A 64-token comparison exactly matched the legacy token sequence and
reduced runtime from 32.7 to 15.8 seconds in the final cold-run matrix. A
320-token stress generated all 320 tokens with resident allocation unchanged at
14,782,134,528 bytes.

## Known limits

- Decode is PCIe/disk expert-miss bound, not shader-compute bound.
- The default tuned configuration uses about 12.8 GiB resident VRAM: 24 expert
  slots per layer plus the resident LM head.
- The MoE compute path uses two stages in one Vulkan submission. Stage 1
  dispatches gate/up rows across the GPU; stage 2 dispatches down-projection
  rows and reduces top-K contributions. A direct resident call improved from
  about 17.2 ms to 1.0 ms with max absolute difference `7.63e-6`.
- Single-token decode RoPE and GQA attention run on CPU because Q/K/V and the KV
  cache are already CPU tensors. At context lengths 8 through 512 this was
  1.2-1.7 times faster than three small Vulkan calls, with max difference below
  `2.3e-8`. Prefill remains on the Vulkan attention path.
- Expert-cache miss staging maps Vulkan buffers on the calling thread, then
  fills the disjoint mapped ranges in parallel C++ tasks before one copy
  submission. Over a 48-token run, measured miss-upload time fell from about
  14.2 seconds to 6.9 seconds while preserving identical tokens.
- Expert choices remain broad over 128 experts; a 24-slot cache reached about a
  54 percent hit rate over a 16-token run.
- Larger cache sizes provide diminishing returns.
- First-token latency is high for prompts selecting many unique experts.
- Naive next-token expert prefetch was rejected after cache-aware simulation:
  speculative uploads outweighed prevented misses.
- The next architectural phase is GPU-resident decode with one submission per
  layer. See [gpu_resident_decode_plan.md](gpu_resident_decode_plan.md).
