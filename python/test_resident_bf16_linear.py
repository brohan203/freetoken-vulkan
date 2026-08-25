"""Validate resident BF16-storage FP32-accumulation linear."""
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

MODEL = pathlib.Path(r"C:\Users\rohanborkar\Downloads\Qwen3-4B")


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
    weights = ShardedSafetensors(MODEL).get(
        "model.layers.0.self_attn.q_proj.weight"
    ).contiguous()
    torch.manual_seed(121)
    x = torch.randn(1, 2560) * 0.02
    reference = x @ weights.float().T
    with ResidentTensor.from_tensor(ext, x) as resident_x:
        with ResidentTensor.empty(ext, (1, weights.shape[0])) as resident_y:
            weight_handle = ext.upload_resident(weights)
            ext.linear_bf16_resident_io(
                resident_x.handle,
                weight_handle,
                0,
                resident_y.handle,
                1,
                weights.shape[0],
                weights.shape[1],
                False,
            )
            actual = resident_y.download()
            ext.free_resident(weight_handle)
    difference = (actual - reference).abs()
    print(
        "shape", tuple(actual.shape),
        "max", difference.max().item(),
        "mean", difference.mean().item(),
        "finite", torch.isfinite(actual).all().item(),
    )
    assert torch.allclose(actual, reference, rtol=1e-4, atol=5e-5)
    print("RESIDENT_BF16_LINEAR_OK")


if __name__ == "__main__":
    main()
