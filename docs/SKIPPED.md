# SKIPPED — running list of unfinished work + shortcuts we took

This is the honest ledger. Every entry here is something we know is missing,
suboptimal, or unverified. Read this before saying "the port is done."

Format: **priority · scope · what · why we skipped**.
Priority: [RED] blocks real usage · [YELLOW] hurts quality/perf · [GREEN] nice-to-have.

---

## Perf / measurement

- [GREEN] · all kernels · **Wall vs GPU timing separation** [DONE] done. `VkQueryPool` timestamps added to `vk_util.hpp` (make_timestamp_query, cmd_reset_and_write_start, cmd_write_end, read_gpu_ms). Adopted in `gemm.exe`. Shows ~0.15 ms queue overhead per dispatch on the 6800 XT — real GPU compute is 5-15% faster than wall for the shapes we care about. Adopt in remaining exes when doing perf work.
- [YELLOW] · GEMM · **No double-buffered LDS loads.** Global fetch and LDS compute are serialized; we could overlap them by ping-ponging two LDS tiles. Expected gain: ~1.5-2× on GEMM at 1024³. Currently at 1198 GFLOPS on 1024³ (5.2% of peak).
- [YELLOW] · GEMM · **No subgroup ops (`VK_KHR_shader_subgroup`).** Cross-lane reductions via `subgroupAdd` / `subgroupMax` are hardware-accelerated on RDNA2 and typically beat shared-memory tree reductions by 5-10× on the reduction step alone.
- [GREEN] · GEMM · **Autotuning.** BM/BN/BK/TM/TN are hardcoded. On small vs. large matrices very different tile sizes win.
- [GREEN] · GEMM · **No M=1 (GEMV) specialization.** GEMV is common (per-token linear projections in decode) and needs a totally different kernel — one workgroup per output element, K reduction across threads.

## Precision / dtypes

- [GREEN] · GEMM · **FP16 storage + FP32 accumulator done for reg_tiled**. shaders/gemm_reg_tiled_f16.comp works, 1.9× faster than F32 at 1024^3 (1466 vs 778 GFLOPS). Same pattern extends to other kernels: enable shaderFloat16 (Vulkan12Features), storageBuffer16BitAccess (Vulkan11Features), add extensions, use float16_t storage, promote to float in inner ops.
- [YELLOW] · all other kernels · **FP16 not yet applied to rmsnorm/softmax/attention/moe_mlp.** Mechanical extension of the reg_tiled_f16 pattern.
- [RED] · all kernels · **No FP8, MXFP4, NVFP4.** FreeToken's headline speed on Blackwell comes from FP4/FP8. RDNA2 has FP16 (which we now support) but not FP8/FP4 in hardware — we'd emulate the packing.
- [YELLOW] · all kernels · **BF16 not supported.** Vulkan requires GL_EXT_bfloat16 extension; some drivers.

## Attention

- [DONE] [GREEN] · attention · **Multi-head + causal mask done** (`flash_attention_mh_f32.comp`). Accepts [B, H, S, D] tensors, optional causal flag. Passes all 7 test shapes including S=1024 causal, max abs err 3e-8. Extension exposes as `flash_attention_mh(Q, K, V, scale, causal=False)`.
- [RED] · `attention_naive` · **Correctness bug at S=512 D=128 specifically.** Non-deterministic. FlashAttention on the same shape always passes so this doesn't block real work.
- [GREEN] · `attention_naive` · **Single head only, single batch.** Superseded by multi-head FA.
- [GREEN] · `attention_naive` · **S ≤ 2048 hard limit.** Superseded by FlashAttention.
- [YELLOW] · `flash_attention` · **No block-Q parallelism (Br = 1).** Standard FA1 uses Br > 1 to amortize KV loads. Costs ~2-3× on long seqs.
- [YELLOW] · `flash_attention` · **BC=32 chosen for LDS budget, not perf.** Needs autotuning.
- [YELLOW] · attention · **No GQA (grouped query attention).** Phi-3.5-MoE has 32 Q heads sharing 8 KV heads.

## MoE

- [DONE] [GREEN] · MoE MLP · **Fused per-token MoE MLP shader done** (`moe_mlp_f32.comp`, small shapes) + **Dff-blocked variant** (`moe_mlp_lg_f32.comp`, up to D=4096). Real Phi-3.5-MoE geometry (D=4096, Dff=6400, E=16, K=2) verified against pure-PyTorch reference, max abs err 2.89e-08.
- [YELLOW] · MoE MLP · **Dff-blocked reads W_down (Dff/BDFF) times per output.** For Phi (Dff=6400, BDFF=512) that's 12.5× redundant global reads. Grouped GEMM would fix.
- [RED] · MoE MLP · **Per-token compute, no grouped GEMM.** Real perf needs to permute tokens by expert assignment, then do E large GEMMs. Our per-token kernel makes each expert's weight matrix re-loaded T*K/E times.
- [RED] · MoE · **No shared-expert path.** Some MoE (DeepSeek-V3) have a "shared expert" every token uses.
- [YELLOW] · router · **Fixed E ≤ 64.** Router shader assumes small expert count.

## PyTorch integration

- [GREEN] · integration · **C++ Torch extension POC done for RMSNorm.** `python/ext_module.cpp` + `build_and_load_ext.py`. Measured **85× speedup vs subprocess** (167 ms → 1.96 ms/call for x[16,2048]). Pattern extends mechanically to other kernels — add init_XXX_pipeline, add op function, add PYBIND11 def.
- [YELLOW] · integration · **Only RMSNorm exposed as C++ op.** Extending to GEMM/attention/MoE etc is copy-and-modify from the RMSNorm entry point. Modest effort per kernel (~50 lines each).
- [YELLOW] · integration · **Buffers alloc/free'd per call.** Persistent buffer pool would amortize the ~0.5ms allocator cost.
- [YELLOW] · integration · **No autograd / backward.** Inference-only.
- [YELLOW] · integration · **No Vulkan-backed torch device.** Weights round-trip through CPU on every call. Full integration = register a custom torch device type + implement its allocator/copy ops.
- [YELLOW] · integration · **Subprocess wrapper still there** for demos and correctness testing. Both wrappers coexist; extension is the fast path.

## Correctness / QA

- [YELLOW] · all kernels · **No Vulkan validation layers enabled at runtime.** They catch use-after-free, bad descriptor bindings, missing barriers, etc. Should turn on in Debug builds.
- [YELLOW] · testing · **No property-based / fuzz testing.** Only a handful of fixed shapes per kernel.
- [GREEN] · testing · **No CI.** Everything's manual.

## Build / distribution

- [YELLOW] · packaging · **No `pip install`-able wheel.** Not shippable to other users.
- [GREEN] · packaging · **No CMake presets or config.** Every new dev has to remember `-G "Visual Studio 17 2022" -A x64`.

## Multi-GPU / distributed

- [GREEN] · future · **No NCCL replacement.** FreeToken's `pynccl.cu` is single-node. RCCL exists for AMD but we haven't touched multi-GPU.

---

## Blocked on user (need help)

None right now. Everything above is engineering debt that we can pay down
autonomously when we get back to it — no external decisions required.

---

## Lessons learned this session (2026-08-22)

- **Coalesced writes matter more than I remembered.** Naive GEMM jumped 11× (29 → 313 GFLOPS) purely by swapping x/y so writes to C were contiguous.
- **Vulkan pipeline creation with too much shared memory returns `VK_ERROR_UNKNOWN` (-13).** Not a friendly error — validation layers would have been nice here. First FlashAttention build hit this from `Ks[64][128]+Vs[64][128]` = 64 KB (exactly the RDNA2 LDS budget). Halved `BC` to 32 → fit.
- **`torch.scaled_dot_product_attention` picks different math paths by shape.** For our attention_naive, S=512 D=128 diverged from SDPA by 1e-3, but explicit `softmax(Q@K^T)V` matched to 1e-8. Always use math-equivalent reference for shader correctness tests.
- **OpenBLAS threadpool blows up on 24-core Windows with default settings.** `OPENBLAS_NUM_THREADS=1` fix baked into venv activate — took an hour to figure out on day 1.
- **Vulkan HKLM env var is only picked up by processes launched AFTER SDK install.** Every subprocess call needs `env["VULKAN_SDK"]` set explicitly if the shell predates the install. Always inject it.
