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

## Next gates

1. Stack all 36 layers with lazy per-layer loading.
2. Run a full-model next-token smoke test.
3. Add KV-cached generation.
4. Compare output tokens against Transformers.
5. Pin BF16/FP32 weights resident only after correctness is established.
