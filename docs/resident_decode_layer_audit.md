# Resident decode layer audit

This audit records the evidence-backed constraints for a fully GPU-resident
single-token gpt-oss decode layer.

## Verified current state

- Resident activation infrastructure exists: allocation, download, RMSNorm,
  residual add, and owned `ResidentTensor` lifecycle.
- Resident projection infrastructure exists: linear I/O, fused RMSNorm plus
  Q/K/V, and fused O projection plus residual plus post-attention norm plus
  router.
- Resident attention infrastructure exists: full-dimension RoPE, capacity-
  strided KV append, attention sinks, GQA, and sliding-window attention.
- Resident router top-4 writes four int32 IDs and four FP32 normalized weights.
- Resident two-stage MoE consumes resident activation/control buffers and
  writes resident output.

## RoPE contract

Both downloaded official checkpoints (`gpt-oss-20b/config.json` and
`gpt-oss-120b/config.json`) omit `partial_rotary_factor` and specify
`head_dim=64` through `rope_parameters`. Full 64-dimension RoPE is the correct
contract for these checkpoints. Earlier partial-RoPE warnings were stale.

## Buffer layout contracts

- Hidden activation: `[T, 2880]`, row-major FP32.
- Q: `[B, 64, S, 64]`, flattened as `[B*64*S*64]`.
- K/V: `[B, 8, S, 64]`.
- Resident KV slabs: `[B, 8, capacity, 64]`; attention must use `capacity` as
  the head stride and `S_kv` as the live length.
- Router logits: `[T, 128]` FP32.
- Router IDs: `[T, 4]` int32.
- Router weights: `[T, 4]` FP32.
- Two-stage MoE hidden scratch: `[T, 4, 2880]` FP32.

## Synchronization requirements

A one-submit layer needs compute write-to-read barriers after:

1. RMSNorm before Q/K/V projections;
2. RoPE before KV append;
3. KV append before attention;
4. attention before O projection;
5. O projection before residual add;
6. residual add before post-attention RMSNorm;
7. post-attention RMSNorm before router;
8. router before top-K;
9. top-K/control remap before MoE;
10. MoE before final residual add.

Host expert-cache management still requires a synchronization point after
resident top-K IDs are produced. Only 16 bytes of IDs need to cross to CPU;
routing weights can remain resident.

## VRAM budget

The tuned 120b baseline already consumes approximately:

- 24 expert slots/layer: about 10.6 GiB;
- resident FP32 LM head: about 2.16 GiB;
- KV cache and transient buffers: remaining headroom.

Pinning all 36 layers of FP32 Q/K/V/O/router weights is not viable. Projection
weights need one of:

1. a bounded layer-weight cache;
2. BF16 resident shaders after a verified bit-layout test;
3. per-layer staging into reusable device-local projection buffers.

## Dispatch guard

The Stage-1 MoE X dispatch is `T * K * ceil(Dff/64)`. With Dff=2880 and K=4,
single-token decode dispatches 180 workgroups and is safe. Large prefill may
exceed Vulkan's 65535 X-dimension limit; the resident I/O path has a runtime
guard and prefill remains on the transient path until a 2D dispatch is added.

## Integration order

1. Keep single-token decode only (`T=1`).
2. Upload one hidden input and keep it resident through the full layer.
3. Download only four global top-K IDs for expert-cache management.
4. Update resident remapped IDs; keep weights resident.
5. Run resident two-stage MoE and final residual.
6. Compare complete layer output against the existing implementation.
7. Only after exact routing and tolerance gates pass, batch all layer phases
   into one command buffer.
