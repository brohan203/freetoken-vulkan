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

## Next gates

1. Add AWQ matrix handles to the resident manager.
2. Pin all 40 layers resident.
4. Run full-model and generation parity.
5. Add long-context stability and CLI examples.
