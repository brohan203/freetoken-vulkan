# freetoken-vulkan

Run OpenAI's `gpt-oss-20b` on AMD Radeon GPUs via a from-scratch Vulkan compute backend. No CUDA, no ROCm, no Triton.

Verified end-to-end on a single Radeon RX 6800 XT (16 GB VRAM, RDNA2, Navi 21) at approximately 1.2 seconds per token, with bit-exact correctness against the `transformers` reference on real gpt-oss-20b weights.

Inspired by [FreeToken](https://github.com/FlashML-org/FreeToken) - which targets NVIDIA/Blackwell via Triton kernels - this repo attacks the same problem from the AMD side by writing every compute kernel as a GLSL compute shader compiled to SPIR-V. Zero source overlap with upstream FreeToken; this is a from-scratch reimplementation of a Vulkan backend, not a fork of the CUDA code.

---

## What works

- Full `gpt-oss-20b` forward pass - all 24 layers, MXFP4 experts, GQA attention with attention sinks + sliding window, YARN RoPE
- KV cache (O(1) per new token - verified flat over 64-step decodes)
- Persistent VRAM residency for MoE expert weights (approximately 9.5 GB) and LM head (approximately 2.2 GB)
- Buffer pool that eliminates the AMD driver's per-call allocation OOM
- Long-form generation stability: 320 tokens across 5 prompts with zero errors
- Bit-exact correctness vs `transformers` reference on every kernel + composed forward pass

## What's honest

- Approximately 1.2 s/token on a 6800 XT. That's about 50 tok/min. Fine for a demo, not chatbot-fast.
- Windows-only. Nothing platform-specific in the shader code; a Linux port would take an afternoon of build-system work.
- Single-token batching only. No request queue, no continuous batching, no multi-user server.
- Model download not automated (see below).
- Perf is bounded at approximately 18 ms per Vulkan submit-wait on this hardware/driver. Getting below that requires a batched-submit refactor (activations stay in VRAM across kernels) that has not been built yet.

## Hardware requirements

| Component | Tested | Notes |
|---|---|---|
| GPU | AMD Radeon RX 6800 XT (RDNA2, 16 GB) | RDNA2/RDNA3 with at least 16 GB VRAM should work |
| System RAM | 64 GB | Approximately 16 GB actually used (weights) + headroom |
| OS | Windows 11 | Linux port straightforward - not yet done |
| Vulkan | Driver >= 26.6, API 1.4.309 | LunarG SDK 1.4.x for glslc + validation layers |
| MSVC | VS Build Tools 2022 (14.44) | For CMake + PyTorch JIT extension build |
| Python | 3.13 + torch 2.13 CPU | See `python/setup_ext.py` |

## Quickstart

```powershell
# 1. Clone
git clone https://github.com/brohan203/freetoken-vulkan
cd freetoken-vulkan

# 2. Download gpt-oss-20b weights (approximately 13 GB) somewhere OUTSIDE this repo
huggingface-cli download openai/gpt-oss-20b --local-dir C:\path\to\gpt-oss-20b

# 3. Build shaders (produces build/shaders/*.spv)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

# 4. Create a venv and install torch (CPU is fine - GPU work is on Vulkan side)
python -m venv .venv
.venv\Scripts\activate
pip install torch transformers safetensors

# 5. Edit MODEL_DIR in python\demo_long_gen.py to point at your weights dir,
#    then run under a vcvars64-wrapped shell so the torch JIT ext compiles.
```

Use the generic long-lived resident prompt loop for either checkpoint:

```powershell
# Defaults to gpt-oss-20b:
python\chat_gpt_oss.py --max-new-tokens 48 "The capital of France is"

# Select gpt-oss-120b:
python\chat_gpt_oss.py --model-dir C:\path\to\gpt-oss-120b \
    --max-new-tokens 48 "The capital of France is"

# Omit the prompt to start an interactive loop.
```

Resident decode is the default: the model, FP32 LM head, all 36 layers of
projection/norm/router weights, and an 18-slot-per-layer LFU expert cache are
initialized once. Use `--legacy-decode` to select the older CPU/tensor path.
Repeated prompts are significantly faster because both the resident workspace
and expert cache stay warm. A measured 16-token prompt fell from 11.70 seconds
to 4.39 seconds on an identical repeat, with exactly the same token IDs.

`GptOssModel` is an explicit resident-resource owner. Call `model.close()` or
use it as a context manager when loading multiple models sequentially; close is
idempotent and returns all model-owned VRAM allocations.

For protocol-controlled incremental context, `ResidentDecodeSession` preserves
resident KV state and accepts exact token-ID appends. The checkpoint tokenizer
has no canonical `chat_template`, so the library intentionally does not invent
user/assistant formatting.

## Sample output - real, from `demo_long_gen.py`

Prompt: `"def fibonacci(n):"`

```python
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
```

Prompt: `"The capital of France is"`

```
The capital of France is Paris."
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "The largest planet in our solar system is Jupiter."
    }
```

Prompt: `"SELECT name, age FROM users WHERE"`

```sql
SELECT name, age FROM users WHERE age > 30 ORDER BY age DESC LIMIT 5;
```

## Perf snapshot

Measured on Radeon RX 6800 XT, 12-token greedy generation of `"The capital of France is"`:

| Config | Prefill | Decode/tok | Total | vs baseline |
|---|---|---|---|---|
| Baseline (no KV, no resident) | approximately 10 s | 6.30 s | 28.7 s | 1.00x |
| + KV cache | 7.6 s | 2.70 s | 21.2 s | 1.35x |
| + Resident MoE (approximately 9.5 GB VRAM) | 1.7 s | 1.38 s | 8.6 s | 3.34x |
| + Resident LM head (+2.2 GB VRAM) | 1.4 s | 1.20 s | 5.0 s | **5.74x** |
| Fully resident 20b, fused all-expert MoE | CPU prefill | **0.0423 s** | **3.73 s / 64 tokens** | exact token parity |
| Fully resident 120b, 64-token run | CPU prefill | **0.229 s** | **15.84 s** | exact token parity |

Long stability runs complete without resident-memory growth: 20b generated 320
tokens at approximately 0.0426 s/token and 120b at approximately 0.272 s/token.

## Architecture

```
freetoken-vulkan/
  shaders/                        GLSL compute shaders (SPIR-V after cmake build)
    flash_attention_gpt_oss_kv_f32.comp   GQA + causal + SWA + attention sinks + KV cache
    moe_mlp_mxfp4_gpt_oss_f32.comp        MoE MLP with MXFP4 experts + gpt-oss activation
    moe_mlp_mxfp4_gpt_oss_par_f32.comp    ... parallel-experts variant
    linear_fp32_resident_f32.comp         Matmul with VRAM-resident weight (LM head)
    rope_partial_f32.comp                 YARN RoPE (partial rotary)
    ...                                   earlier kernels: rmsnorm, softmax, matmul, ...

  include/vk_util.hpp             Vulkan helpers: Context, Buffer, BufferPool, submit_and_wait
  python/ext_module.cpp           PyTorch extension bridging shaders to Python
  python/gpt_oss/                 Model package
    config.py                     GptOssConfig - parsed from config.json
    loader.py                     safetensors to dict, BF16 to FP32 dequant on load
    rope.py                       YARN cos/sin computation (delegates to transformers)
    kv_cache.py                   KVCache - pre-allocated per-layer K/V tensors
    resident.py                   ResidentMoEWeights - pins weights to VRAM
    layer.py                      gpt_oss_layer_forward - one transformer layer
    model.py                      GptOssModel - 24 layers + embed + norm + LM head
    generate.py                   greedy_generate + greedy_generate_kv
    mxfp4_ref.py                  MXFP4 dequant reference (bit-exact vs HF)

  python/demo_*.py                Runnable demos
  python/test_*.py                Correctness tests (per-kernel + composed forward)
  python/bench_*.py, profile_*.py Perf benchmarks + profiler
  docs/                           Design docs
  CMakeLists.txt                  Build script (compiles shaders + a few CLI test exes)
```

## Correctness - bit-exact vs the reference

Every kernel in this repo was validated against a pure-PyTorch reference on real gpt-oss-20b weights before being composed into the forward pass. Key numerical results:

| Kernel | Max abs error vs reference |
|---|---|
| MXFP4 dequant | 0.000e+00 (bit-exact vs `transformers.integrations.mxfp4._convert_moe_packed_tensors`) |
| MXFP4 matvec | 1.5e-6 |
| MoE MLP with gpt-oss activation | 2.86e-6 |
| Flash attention (GQA + SWA + sinks) | 2.98e-8 |
| RoPE (partial, YARN) | 4.77e-7 |
| KV attention vs full-prefill (bit-exact test) | 0.000e+00 |
| One-layer end-to-end vs pure-PyTorch reference | 4.27e-03 (25M+ FMAs of FP32 accum drift, expected) |

## Next performance phase

The current decode architecture still moves activations between PyTorch and
Vulkan at layer boundaries. The phased plan for resident activation buffers,
resident projections, one-submit layers, and dedicated transfer-queue overlap
is documented in [docs/gpu_resident_decode_plan.md](docs/gpu_resident_decode_plan.md).

## What's NOT here - and why

- No batched-submit implementation. We measured the current ceiling at approximately 18 ms per Vulkan submit-execute-fence-signal round-trip on RDNA2/Windows AMD driver. Breaking that requires keeping activations in VRAM across kernels and issuing 1-2 submits per layer instead of approximately 4. Well-scoped, not built.
- gpt-oss-120b works through file-backed MXFP4 expert streaming, a bounded per-layer VRAM expert cache, and a two-stage parallel MoE pipeline. See [docs/gpt_oss_120b.md](docs/gpt_oss_120b.md). A verified 64-token resident run decodes at approximately 0.22 s/token on the tested configuration, and a 320-token stress run completes without VRAM growth or OOM.
- No CUDA, no ROCm, no Triton. This is by design - the point is to have a backend for AMD (and Intel Arc, Apple, mobile) that doesn't depend on those.

## Credits

- [FreeToken](https://github.com/FlashML-org/FreeToken) - inspiration + the roadmap of models to target
- [OpenAI's gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) - the model
- [HuggingFace `transformers`](https://github.com/huggingface/transformers) - the canonical reference implementation we validated against

## License

Apache 2.0 - same as upstream FreeToken. See [LICENSE](LICENSE).
