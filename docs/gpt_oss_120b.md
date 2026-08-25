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
| 28-slot GPU expert cache | 5 / 16 | 9.8 s | 2.9 s/token | 46.1 s |

Twenty-four slots are the default because 28 slots consume more VRAM for only a
small gain. Resident LM head was also tested; it fit alongside the expert cache
but did not improve the cold-heavy short benchmark enough to justify enabling
it by default.

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
FREETOKEN_GPU_CACHE
FREETOKEN_PIN_LM_HEAD
FREETOKEN_PREFILL_CHUNK
```

Recommended defaults:

```text
FREETOKEN_CACHE_SLOTS=24
FREETOKEN_GPU_CACHE=1
FREETOKEN_PIN_LM_HEAD=0
FREETOKEN_PREFILL_CHUNK=0
```

## Known limits

- Decode is PCIe/disk expert-miss bound, not shader-compute bound.
- Expert choices remain broad over 128 experts; a 24-slot cache reached about a
  54 percent hit rate over a 16-token run.
- Larger cache sizes provide diminishing returns.
- First-token latency is high for prompts selecting many unique experts.
- An asynchronous prefetch pipeline could hide part of the expert-read and
  upload latency, but is not implemented.
