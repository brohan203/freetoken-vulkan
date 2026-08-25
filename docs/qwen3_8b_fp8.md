# Qwen3-8B FP8 Runtime

## Checkpoint status

The official `Qwen/Qwen3-8B-FP8` checkpoint is downloaded and validated locally.
It uses block-scaled FP8 E4M3 weights and FP32 inverse scales.

- 8.803 GiB local checkpoint directory
- 9 safetensors shards
- 470 indexed tensors
- zero missing shards
- weight blocks: `128 x 128`
- weight format: `float8_e4m3fn`
- scale tensors: FP32 `weight_scale_inv`

The BF16 Qwen3-8B checkpoint would require approximately 15.28 GiB for weights
plus a compact workspace, exceeding the empirically safe resident allocation
budget on this Windows Vulkan driver. The official FP8 checkpoint fits.

## FP8 resident linear

`linear_fp8e4m3_resident_f32.comp` stores one FP8 byte per weight and reads one
FP32 scale per 128x128 weight block. Activations, accumulation, and output remain
FP32.

The verified dequantization contract matches Transformers:

```text
dequantized_weight[row, col] =
    fp8_e4m3(weight[row, col]) * scale[row // 128, col // 128]
```

A real layer-0 Q projection (`4096 x 4096`) produced:

- scale grid: `32 x 32`
- maximum absolute error versus PyTorch dequantization: `5.960e-8`
- mean absolute error: `1.050e-8`
- finite FP32 output

## Next gates

1. Extend the Qwen loader with FP8 weight/scale pairs.
2. Pin all Qwen3-8B FP8 layers resident.
3. Validate one resident layer against Transformers dequantized math.
4. Run full-model logits and greedy-generation parity.
5. Benchmark sustained generation and memory stability.
