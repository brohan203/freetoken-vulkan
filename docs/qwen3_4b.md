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

## Next gates

1. Add KV-cached generation.
2. Validate multi-token output against Transformers greedy generation.
3. Pin BF16/FP32 weights resident only after generation correctness passes.
4. Replace CPU dense projections with resident Vulkan operations.
5. Add a Qwen3 CLI after sustained-generation stability is verified.
