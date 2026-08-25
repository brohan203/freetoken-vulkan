"""Validate packed AWQ4 resident linear on Qwen3-14B."""
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

MODEL = pathlib.Path(r"C:\Users\rohanborkar\Downloads\Qwen3-14B-AWQ")
ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


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
    qweight = tensors.get(prefix + ".qweight").contiguous()
    qzeros = tensors.get(prefix + ".qzeros").contiguous()
    scales = tensors.get(prefix + ".scales").contiguous()
    k_size = qweight.shape[0]
    n_size = scales.shape[1]
    weights = torch.empty((k_size, n_size))
    zeros = torch.empty((k_size, n_size))
    packed_weights = qweight.to(torch.int64)
    packed_zeros = qzeros.to(torch.int64)
    for packed in range(qweight.shape[1]):
        for slot, logical in enumerate(ORDER):
            column = packed * 8 + logical
            weights[:, column] = (
                (packed_weights[:, packed] >> (4 * slot)) & 15
            ).float()
            zero_column = (
                (packed_zeros[:, packed] >> (4 * slot)) & 15
            ).float()
            zeros[:, column] = zero_column.repeat_interleave(128)[:k_size]
    expanded_scales = scales.float().repeat_interleave(128, 0)[:k_size]
    dequantized = (weights - zeros) * expanded_scales
    torch.manual_seed(161)
    x = torch.randn(1, k_size) * 0.02
    reference = x @ dequantized

    with ResidentTensor.from_tensor(ext, x) as resident_x:
        with ResidentTensor.empty(ext, (1, n_size)) as resident_y:
            weight_handle = ext.upload_resident(qweight)
            zeros_handle = ext.upload_resident(qzeros)
            scales_handle = ext.upload_resident(scales)
            ext.linear_awq4_resident_io(
                resident_x.handle,
                weight_handle,
                zeros_handle,
                scales_handle,
                resident_y.handle,
                1,
                n_size,
                k_size,
                128,
            )
            actual = resident_y.download()
            ext.free_resident(weight_handle)
            ext.free_resident(zeros_handle)
            ext.free_resident(scales_handle)

    difference = (actual - reference).abs()
    print(
        "shape", tuple(actual.shape),
        "qweight", tuple(qweight.shape),
        "qzeros", tuple(qzeros.shape),
        "scales", tuple(scales.shape),
        "max", difference.max().item(),
        "mean", difference.mean().item(),
        "finite", torch.isfinite(actual).all().item(),
    )
    assert torch.allclose(actual, reference, rtol=2e-4, atol=1e-4)
    print("RESIDENT_AWQ4_LINEAR_OK")


if __name__ == "__main__":
    main()
