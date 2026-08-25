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

## Full model and resident generation

The shared Qwen loader now detects FP8 matrices and dequantizes them lazily for
the correctness path. The resident path stores raw FP8 matrices and their FP32
scale grids directly.

Verified full-model smoke:

- finite logits shape `[1, 1, 151936]`
- top prediction: ` Paris`
- lazy full forward: `10.96 s`
- mean lazy layer time: `0.292 s`

Verified fully resident model:

- all 36 FP8 layers resident
- BF16 embeddings and LM head resident
- resident allocation: `9,457,838,720` bytes (about 8.81 GiB)
- pin time: `5.82-5.99 s`
- one full resident token: `0.0977 s`
- top-10 IDs exactly match the lazy dequantized model
- maximum logit difference: `5.53e-5`
- mean logit difference: `1.27e-5`

Eight-token resident generation exactly matches the lazy reference:

```text
The capital of France is Paris. The capital of Italy is Rome
```

Performance:

- lazy generation: `81.92 s`
- fused resident generation: `0.521 s`
- prompt processing: `0.0446 s/token`
- decode: `0.0418 s/token` (about 23.9 tokens/second)
- resident allocation unchanged before/after generation
- explicit cleanup returns resident bytes exactly to zero

## Fused layer and long stability

The same fused dense-layer operation now handles both native BF16 and
block-scaled FP8 weights by switching the resident linear pipeline and binding
FP8 scale grids. Eight-token parity remains exact.

A forced 320-token run at max sequence 384 completed with:

- total runtime: `15.44 s`
- decode average: `0.0467 s/token`
- steady resident allocation: `9,552,210,560` bytes
- identical resident bytes before and after generation
- successful explicit cleanup

## CLI

The shared Qwen CLI accepts the 8B FP8 checkpoint directly:

```powershell
python\chat_qwen3.py \
    --model-dir C:\path\to\Qwen3-8B-FP8 \
    --max-new-tokens 48 "The capital of France is"
```

Verified CLI smoke:

- pin: `9.34 s`
- resident allocation reported: `8.90 GiB`
- prompt/decode output: `The capital of France is Paris.`
- prompt processing: `0.047 s/token`
- decode: `0.042 s/token`

## Next gates

1. Benchmark larger context capacities.
2. Add batch-size and continuous-batching support.
