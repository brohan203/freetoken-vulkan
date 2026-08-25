# Qwen3-14B AWQ Runtime

## Checkpoint status

The official `Qwen/Qwen3-14B-AWQ` checkpoint is downloaded and validated
locally.

- 9.304 GiB indexed payload
- 1,003 tensors
- 2 safetensors shards
- zero missing or extra indexed tensors
- 40 layers
- hidden size 5120
- intermediate size 17408
- 40 query heads / 8 KV heads
- head dimension 128

Dense matrices use packed 4-bit AWQ:

- `qweight`: INT32 `[input, output / 8]`
- `qzeros`: INT32 `[input / 128, output / 8]`
- `scales`: BF16 `[input / 128, output]`
- group size: 128
- zero point enabled

Embeddings, norm vectors, and LM head remain BF16.

## Verified AWQ contract

The packed layout was derived empirically against the official BF16
`Qwen/Qwen3-14B` layer-0 Q projection. The unique high-correlation convention
is:

- packed nibble logical order `[0, 2, 4, 6, 1, 3, 5, 7]`
- no additional `+1` zero-point offset

This convention has correlation `0.9935` with the official BF16 tensor;
natural/inverse alternatives are only about `0.24`.

## Resident AWQ4 linear

`linear_awq4_resident_f32.comp` reads packed INT32 weights/zeros and BF16 scales
directly from resident buffers, then accumulates/outputs FP32.

A real layer-0 Q projection produced:

- output shape `[1, 5120]`
- maximum error versus exact PyTorch unpack: `5.960e-8`
- mean error: `8.208e-9`
- finite FP32 output

## Lazy dense-layer loader

The shared Qwen loader now recognizes AWQ matrix triplets, unpacks them with the
verified nibble order, applies group-128 scales/zeros, and transposes AWQ's
`[input, output]` layout into the dense runtime's `[output, input]` contract.

One full real Qwen3-14B-AWQ layer compared with an independent PyTorch reference:

- output shape `[1, 3, 5120]`
- maximum absolute error: `2.861e-6`
- mean absolute error: `5.223e-8`
- finite output

## Fully resident model

The resident manager stores every packed qweight/qzeros/scales triplet without
dequantizing it. Norms, embeddings, and LM head remain BF16/FP32 as in the
checkpoint.

Full-model gate:

- all 40 layers resident
- resident allocation: `9,999,452,832` bytes (about 9.31 GiB)
- pin time: `6.20 s`
- one full resident token: `0.926 s`
- top-10 IDs exactly match the lazy dequantized model
- maximum logit difference: `1.097e-5`
- mean logit difference: `2.194e-6`
- explicit cleanup returns resident bytes exactly to zero

The lazy one-token reference took `702.9 s` because all 40 layers had to be
unpacked into FP32 sequentially.

Eight-token resident generation:

```text
The capital of France is Paris. What is the capital of the
```

- prompt processing: `0.920 s/token`
- decode: `0.918 s/token`
- steady resident bytes unchanged at `9,999,452,832`

This initial path is intentionally unfused. Correctness and resident fit are
established before AWQ layer command fusion.

## Next gates

1. Fuse AWQ dense-layer submissions.
2. Run a 320-token resident stability test.
3. Add CLI examples.
