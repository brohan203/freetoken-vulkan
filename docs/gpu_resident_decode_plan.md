# GPU-resident decode pipeline plan

## Goal

Reduce gpt-oss-120b decode below the current verified approximately 0.394
seconds/token by eliminating CPU/Vulkan activation round-trips and per-op queue
submission waits.

This is a staged refactor. Every phase must preserve the existing CPU/tensor
path as a reference and pass the 20b and 120b regression matrix before the next
phase starts.

## Current measured baseline

- 48-token total: approximately 22.8 seconds
- Decode: approximately 0.394 seconds/token
- Expert upload: approximately 6.9 seconds over 48 tokens
- Expert materialization: negligible through mmap-backed rows
- Two-stage resident MoE: approximately 1 ms for a warm top-4 invocation
- CPU single-token attention is faster than three small Vulkan submissions
- Remaining costs: expert miss transfer, CPU projections, and layer orchestration

## Phase 1: resident activation handles

Add a typed Python/C++ handle for device-local FP32 activation buffers.

Required operations:

1. allocate activation `[T, D]`;
2. upload embedding output once per token;
3. download final hidden state only when needed;
4. resident RMSNorm;
5. resident residual add;
6. explicit buffer barriers between operations.

Correctness gate:

- random RMSNorm/residual tensors match PyTorch within `1e-5`;
- one real 20b layer matches the existing path;
- no change to QKV, attention, router, or MoE yet.

Memory budget:

- two hidden buffers: `2 * 2880 * 4` bytes per decode token;
- temporary normalized buffer: `2880 * 4` bytes;
- negligible relative to weight caches.

### Phase 1 status

Implemented and validated:

- owned `ResidentTensor` FP32 handles;
- resident download through a pooled transfer buffer;
- handle-based resident RMSNorm;
- handle-based resident residual add;
- explicit free/context-manager lifecycle and allocation accounting.

Correctness:

- random RMSNorm max error `4.77e-7`;
- resident add is bit-exact against its resident input;
- real 20b activation RMSNorm max error `1.91e-6`;
- allocation counter returns to baseline after explicit free.

Isolated RMSNorm + add remains slower than CPU (`0.209 ms` vs `0.063 ms`)
because it pays two submissions. Phase 2/3 must batch resident operations for the
infrastructure to improve model latency.

## Phase 2: resident projections and router

Move decode-time Q, K, V, O, and router matvecs onto Vulkan.

Weight storage strategy:

- keep checkpoint weights in BF16 to avoid doubling VRAM;
- decode BF16 in shader using packed 32-bit loads;
- do not use the rejected BF16 LM-head implementation until a small-vector
  bit-layout test is added and passes;
- Q/K/V/O weights are processed one layer at a time or cached in a bounded
  layer-weight cache because all 36 layers do not fit alongside experts.

Outputs:

- Q `[64, 64]`;
- K/V `[8, 64]`;
- router logits `[128]`;
- top-K IDs and weights remain resident until expert dispatch.

Correctness gate:

- each projection matches BF16-to-FP32 PyTorch reference;
- router top-4 IDs are identical on a recorded 120b trace;
- one-layer final output remains within established FP32 tolerance.

### Phase 2 foundation status

Implemented and validated:

- handle-based resident FP32 linear projection;
- fused resident RMSNorm + Q/K/V projection in one submission;
- explicit compute write-to-read barrier between normalization and matvecs;
- real 20b projection weights and activations.

Correctness and performance:

- resident RMSNorm + Q projection max error `2.29e-5`;
- fused Q/K/V max errors `3.05e-5`, `3.43e-5`, and `2.48e-5`;
- fused resident RMSNorm + Q/K/V: `0.262 ms`;
- equivalent CPU chain at 12 threads: `1.777 ms`;
- local speedup approximately 6.8x.

The next Phase 2 step is resident O projection + residual, post-attention
RMSNorm, and router projection/top-K before full layer batching.

## Phase 3: one submit per layer

Record a layer command buffer containing:

1. RMSNorm;
2. Q/K/V projection;
3. RoPE;
4. attention;
5. O projection + residual;
6. post-attention RMSNorm;
7. router + top-K;
8. expert-cache update barrier;
9. two-stage MoE;
10. final residual.

The layer waits once rather than once per operation.

For single-token decode, CPU attention should remain available as a crossover
fallback until the fully batched Vulkan layer beats it end-to-end.

Correctness gate:

- recorded prompt produces identical token IDs for at least 64 tokens;
- 20b and 120b layer tests pass;
- no new VRAM OOM over 320 generated tokens.

## Phase 4: transfer-queue overlap

Hardware query on the tested RX 6800 XT found:

- family 0: 8 graphics/compute/transfer queues;
- family 1: 4 compute/transfer queues;
- family 2: 2 dedicated transfer-only queues.

After router/top-K are resident, submit expert misses on the transfer queue and
signal a timeline semaphore. The compute queue waits only immediately before
MoE, allowing any remaining attention/O/router work to overlap when possible.

Do not implement speculative next-token prefetch: cache-aware simulation found
poor precision and excess PCIe traffic.

## Completion target

A realistic first target is below 0.25 seconds/token on 120b without changing
model precision or token output. A stretch target is below 0.15 seconds/token
if layer submission batching and transfer overlap both deliver.
