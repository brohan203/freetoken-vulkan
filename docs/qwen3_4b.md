# Qwen3-4B Dense Runtime

## Status

The official `Qwen/Qwen3-4B` BF16 checkpoint has been downloaded and validated
locally. The first implementation milestone supports loading and executing one
real dense transformer layer with reusable Vulkan primitives.

Checkpoint integrity:

- 7.507 GiB on disk
- 7.492 GiB indexed tensor payload
- 398 tensors
- 3 safetensors shards
- zero missing or extra indexed tensors

## Architecture

The verified checkpoint uses:

- 36 decoder layers
- hidden size 2560
- intermediate size 9728
- 32 query heads
- 8 key/value heads
- head dimension 128
- full-dimension RoPE with theta 1,000,000
- per-head Q and K RMSNorm after projection
- bias-free Q/K/V/O projections
- dense `silu(gate) * up` SwiGLU
- tied embedding and LM-head weights
- vocabulary size 151,936

## Implemented contracts

`python/model_contracts.py` defines architecture-neutral dense decoder config
and layer-weight dataclasses. The separate `python/qwen3/` package provides:

- strict Qwen3 config parsing and shape validation
- sharded safetensors loading
- BF16-to-FP32 weight conversion
- standard Qwen3 RoPE table generation
- one dense transformer layer

The gpt-oss runtime remains independent and unchanged.

## Reused Vulkan operations

The layer reuses existing kernels for:

- global RMSNorm
- per-head Q/K RMSNorm
- full RoPE
- grouped-query causal attention without sinks
- SwiGLU activation

Linear projections are CPU FP32 in the initial correctness milestone. Resident
weight and projection optimization starts only after full-model parity passes.

## Correctness gate

`python/test_qwen3_layer.py` compares one real layer against an independent
pure-PyTorch implementation using the same checkpoint weights.

Verified result for input shape `[1, 3, 2560]`:

- output shape exact
- maximum absolute error: `9.5367431640625e-7`
- mean absolute error: `5.0243215810041875e-8`
- all outputs finite
- tolerance gate passed

## Full-model status

`Qwen3Model` now stacks all 36 layers while materializing only one FP32 layer
at a time. On the prompt `The capital of France is`:

- full forward time: `2.91-2.96 s`
- finite logits shape: `[1, 1, 151936]`
- top prediction: ` Paris`
- mean layer time: `54.7 ms`
- maximum layer time: `137.6 ms`
- process working set after touching all mapped shards: `7.93 GiB`

### Transformers parity

A full-model comparison against the official Transformers Qwen3 implementation
using BF16 CPU math produced:

- identical top-10 token IDs in identical order
- identical top-1 prediction (` Paris`)
- top-5 overlap: 5/5
- maximum raw-logit difference: `0.2861`
- mean raw-logit difference: `0.0706`
- Transformers forward: `2.30 s`
- Vulkan-backed milestone forward: `2.96 s`

Raw-logit differences are expected because Transformers accumulates from BF16
weights while this correctness runtime expands each layer to FP32. Token ranking
parity is the acceptance criterion for this milestone.

## KV-cached generation

The dense runtime now uses a preallocated architecture-neutral FP32 KV cache.
Prefill writes rotated K/V for every layer; single-token decode appends one row
and uses grouped-query attention over the cached prefix.

One-layer cache equivalence for a three-token prefill plus one decode token:

- maximum absolute difference versus one full causal pass: `6.184e-7`
- mean absolute difference: `1.103e-7`

Eight-token greedy generation is exactly token-identical to Transformers:

```text
The capital of France is Paris. The capital of Germany is Berlin
```

Token IDs:

```text
[12095, 13, 576, 6722, 315, 9856, 374, 19846]
```

Initial correctness-path performance:

- prefill: `2.99 s`
- decode: `2.20 s/token`
- eight-token runtime: `18.41 s`
- Transformers eight-token runtime: `5.21 s`
- KV cache at max sequence 64: `18,874,368` bytes

This path still reloads every layer and performs CPU FP32 dense projections on
every token. The performance gap is expected until weights and projections are
resident.

## Native BF16 resident linear

A new Vulkan linear kernel stores checkpoint weights in native BF16 and
reconstructs each value exactly for FP32 accumulation and output. This avoids
both FP32's 2x VRAM expansion and FP16 conversion drift.

A real layer-0 Q projection (`4096 x 2560`) was compared with CPU FP32:

- maximum absolute error: `5.029e-8`
- mean absolute error: `6.439e-9`
- output shape and finiteness exact

This primitive makes full Qwen3-4B weight residency feasible within 16 GiB.

## Resident dense layer

The first fully resident Qwen3 decode layer now uses:

- native BF16 Q/K/V/O/gate/up/down weights
- FP32 global and per-head RMSNorm
- resident full RoPE and KV append
- resident grouped-query attention
- resident SwiGLU
- resident residuals and outputs

A real layer-0 decode step compared with the verified cached path produced:

- maximum absolute error: `1.431e-6`
- mean absolute error: `9.456e-8`
- one-layer weights/workspace/KV resident allocation: `206,841,984` bytes
- resident bytes after explicit free: exactly zero

## Fully resident model and generation

All Qwen3-4B weights now remain resident:

- native BF16 embeddings and dense matrices
- FP32 RMSNorm vectors
- native BF16 tied LM head
- FP32 activations, KV cache, attention, and logits

The complete model plus a max-sequence-64 workspace uses
`8,065,071,744` resident bytes (about 7.51 GiB).

Eight-token generation remains exactly token-identical to Transformers:

```text
The capital of France is Paris. The capital of Germany is Berlin
```

Measured resident performance:

- model pin time in the validated warm run: `6.10 s`
- complete five-token prompt plus eight generated tokens: `0.975 s`
- prompt processing: `0.0820 s/token`
- decode: `0.0798 s/token` (about 12.5 tokens/second)
- resident allocation unchanged before/after generation
- explicit workspace/model cleanup returns resident bytes exactly to zero

This is approximately 27x faster per decode token than the lazy correctness
path (`2.20 s/token`).

## Next gates

1. Add sustained-generation stability tests.
2. Add a Qwen3 CLI.
3. Fuse resident dense-layer submissions for additional decode throughput.
4. Benchmark longer contexts and workspace capacities.
