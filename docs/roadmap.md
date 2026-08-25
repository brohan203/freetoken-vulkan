# Roadmap

## Phase 1 — Bootstrap  [DONE] done

`vector_add.exe` prints `PASS` on the 6800 XT. 1.6 ms for 1M FP32 additions.

## Phase 2 — RMSNorm  [DONE] done

Max abs diff vs `torch.rms_norm` = 1e-6 across 6 shapes. Introduces workgroup
reduction + shared memory.

## Phase 3 — GEMM  [DONE] done, 3 variants

- 3.1 naive:      313 GFLOPS on 1024³ (after coalescing fix)
- 3.2 tiled LDS:  695 GFLOPS on 1024³
- 3.3 reg tiled:  736 GFLOPS on 1024³, 12.8 GFLOPS on GEMV

~3% of the 6800 XT's theoretical FP32 peak. See SKIPPED.md for perf debt
(no double-buffering, no subgroup ops, no autotune, no M=1 specialization).

## Phase 4 — Softmax + Attention  [DONE] done

- 4.1 softmax: numerically stable (handles inputs at magnitude 50)
- 4.2 fused single-head attention: passes vs torch reference, S ≤ 2048

## Phase 5 — FlashAttention v1  [DONE] done

Online-softmax attention with block-KV streaming. Br=1 (see SKIPPED for Br>1).

Passes at S ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192} — max err 6e-9.
No LDS-imposed sequence length ceiling.

## Phase 6 — MoE building blocks  [DONE] done (correctness), scaling TODO

- 6.1 `moe_router_f32.comp` — top-K softmax router with renormalization [DONE]
- 6.2 `swiglu_f32.comp` — silu(gate) * up elementwise [DONE]
- 6.3 `moe_mlp_f32.comp` — **fused per-token MoE MLP** with SwiGLU + weighted expert accumulation [DONE] (all 7 test shapes pass, max err 1.2e-8; D ≤ 256, Dff ≤ 512 due to LDS)

## Phase 7 — PyTorch integration  [YELLOW] POC done

- `python/freetoken_vulkan.py` — Python wrapper via subprocess IPC. All 7 kernel entry points exposed. Correct but slow (~130 ms/kernel overhead).
- `demo/demo_mini_transformer.py` — 11-kernel dense transformer block, matches PyTorch to 2.5e-6 abs.
- `demo/demo_moe_transformer.py` — **12-kernel MoE transformer block**, matches PyTorch to 1.6e-6 abs. Full attention + routing + fused MoE MLP.
- **Real C++ torch extension — NOT YET.** See SKIPPED.

## Phase 8 — Perf infrastructure  [DONE] done

- `VkQueryPool` timestamp helpers added to `vk_util.hpp`
- `gemm.exe` reports both wall-clock and GPU-only timing
- Revealed we're actually at **1.2 TFLOPS on 1024³ FP32 GEMM** — ~5.2% of RDNA2 peak

## Phase 9 — FP16 mixed-precision  [DONE] POC done

- `gemm_reg_tiled_f16.comp` — FP16 storage + FP32 accumulator
- Enabled Vulkan features: `shaderFloat16` + `storageBuffer16BitAccess` (in `vk_util.hpp::create_context`)
- **1466 GFLOPS on 1024³ vs 778 GFLOPS FP32** — 1.9× win, matching memory-bandwidth-bound expectation
- Pattern established; other kernels can adopt the same recipe

## Phase 10 — C++ Torch extension  [DONE] full op set done

- `python/ext_module.cpp` — PYBIND11 module registering all 8 ops as
  Vulkan-backed torch functions: `rmsnorm`, `matmul`, `softmax`,
  `flash_attention`, `flash_attention_mh` (multi-head + causal),
  `swiglu`, `moe_router`, `moe_mlp`, `moe_mlp_lg`.
- `python/build_and_load_ext.py` — JIT-compile using `torch.utils.cpp_extension.load()`
- `python/_build_ext.bat` — vcvars wrapper (required on Windows)
- **Extension speeds: ~1-2 ms per kernel call.**
- **`demo/demo_fast_transformer.py`** — dense + MoE transformer blocks
  end-to-end via the extension. Matches PyTorch reference to 2e-6.
  Dense block: **21 ms wall via extension vs 3831 ms subprocess (183×)**.
  MoE block: 24 ms via extension vs 3392 ms subprocess (144×).

## Phase 11 — Multi-head + causal attention  [DONE] done

- `shaders/flash_attention_mh_f32.comp` — extends FA with (B, H) dimension
  in dispatch and optional causal masking. Passes 7 shapes including S=1024
  causal, max abs err 3e-8.
- Extension binding: `flash_attention_mh(Q, K, V, scale, causal=False)`.

## Phase 12 — Dff-blocked MoE MLP (real Phi shape)  [DONE] done

- `shaders/moe_mlp_lg_f32.comp` — blocks along Dff so we can handle real
  MoE model sizes without exceeding 64 KB LDS.
- **Verified at Phi-3.5-MoE geometry** (D=4096, Dff=6400, E=16, K=2)
  vs pure-PyTorch reference — max abs err 2.89e-08.
- Extension binding: `moe_mlp_lg(x, indices, weights, W_gate, W_up, W_down)`.

## Later — perf work, other GPUs, upstream

- Radeon GPU Profiler passes on hot kernels
- Test on Intel Arc (Xe cores)
- MoltenVK for Apple Silicon
- Open PR against `FlashML-org/FreeToken` proposing the backend

## What we intentionally skipped

- FP8 / FP4 quant kernels — last thing to add
- Multi-GPU / distributed (NCCL replacement)
- Training kernels (autograd, gradients) — inference only for now
