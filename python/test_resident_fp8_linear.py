"""Validate resident block-scaled FP8 E4M3 linear on Qwen3-8B."""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = (
    os.path.join(os.environ["VULKAN_SDK"], "Bin")
    + os.pathsep
    + os.environ.get("PATH", "")
)

import torch
from torch.utils.cpp_extension import load

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from gpt_oss import ResidentTensor
from qwen3 import ShardedSafetensors

MODEL = pathlib.Path(r"C:\Users\rohanborkar\Downloads\Qwen3-8B-FP8")


def main() -> None:
    sdk = pathlib.Path(os.environ["VULKAN_SDK"])
    ext = load(
        name="freetoken_vulkan_ext",
        sources=[str(HERE / "ext_module.cpp")],
        extra_include_paths=[str(sdk / "Include"), str(REPO / "include")],
        extra_ldflags=[f"/LIBPATH:{sdk / 'Lib'}", "vulkan-1.lib"],
        extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
        verbose=False,
    )
    tensors = ShardedSafetensors(MODEL)
    prefix = "model.layers.0.self_attn.q_proj"
    weights = tensors.get(prefix + ".weight").contiguous()
    scales = tensors.get(prefix + ".weight_scale_inv").float().contiguous()
    expanded_scales = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    expanded_scales = expanded_scales[: weights.shape[0], : weights.shape[1]]
    torch.manual_seed(141)
    x = torch.randn(1, weights.shape[1]) * 0.02
    reference = x @ (weights.float() * expanded_scales).T

    with ResidentTensor.from_tensor(ext, x) as resident_x:
        with ResidentTensor.empty(ext, (1, weights.shape[0])) as resident_y:
            weight_handle = ext.upload_resident(weights)
            scales_handle = ext.upload_resident(scales)
            ext.linear_fp8e4m3_resident_io(
                resident_x.handle,
                weight_handle,
                scales_handle,
                resident_y.handle,
                1,
                weights.shape[0],
                weights.shape[1],
                scales.shape[1],
            )
            actual = resident_y.download()
            ext.free_resident(weight_handle)
            ext.free_resident(scales_handle)

    difference = (actual - reference).abs()
    print(
        "shape", tuple(actual.shape),
        "dtype", weights.dtype,
        "scales", tuple(scales.shape),
        "max", difference.max().item(),
        "mean", difference.mean().item(),
        "finite", torch.isfinite(actual).all().item(),
    )
    assert torch.allclose(actual, reference, rtol=2e-4, atol=1e-4)
    print("RESIDENT_FP8_LINEAR_OK")


if __name__ == "__main__":
    main()
