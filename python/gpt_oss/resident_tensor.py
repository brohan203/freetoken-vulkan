"""Owned device-local FP32 tensor handles for the Vulkan backend."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable

import torch


@dataclass
class ResidentTensor:
    ext: object
    handle: int
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    _freed: bool = False

    @classmethod
    def from_tensor(cls, ext, tensor: torch.Tensor) -> "ResidentTensor":
        if tensor.dtype != torch.float32:
            raise TypeError("ResidentTensor currently supports float32 only")
        contiguous = tensor.contiguous()
        return cls(ext, ext.upload_resident(contiguous), tuple(contiguous.shape))

    @classmethod
    def empty(
        cls, ext, shape: Iterable[int], dtype: torch.dtype = torch.float32
    ) -> "ResidentTensor":
        shape = tuple(int(dimension) for dimension in shape)
        if dtype != torch.float32:
            raise TypeError("ResidentTensor currently supports float32 only")
        if any(dimension < 0 for dimension in shape):
            raise ValueError("ResidentTensor dimensions must be non-negative")
        bytes_required = prod(shape) * torch.tensor([], dtype=dtype).element_size()
        if bytes_required <= 0:
            raise ValueError("ResidentTensor allocation must be non-empty")
        return cls(ext, ext.allocate_resident(bytes_required), shape, dtype)

    @property
    def numel(self) -> int:
        return prod(self.shape)

    def download(self) -> torch.Tensor:
        self._require_live()
        return self.ext.download_resident(self.handle, list(self.shape))

    def free(self) -> None:
        if self._freed:
            return
        self.ext.free_resident(self.handle)
        self._freed = True

    def _require_live(self) -> None:
        if self._freed:
            raise RuntimeError("ResidentTensor has been freed")

    def __enter__(self) -> "ResidentTensor":
        self._require_live()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.free()

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass
