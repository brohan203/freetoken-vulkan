"""mxfp4_ref.py — reference MXFP4 dequantization in pure PyTorch.

MXFP4 (Open Compute Project microscaling standard, 2023):
    - Weight element: E2M1 (1 sign + 2 exponent + 1 mantissa bits) → 4 bits
    - Scale per block of 32 elements: E8M0 (unsigned biased exponent) → 8 bits
    - Effective bits per weight ≈ 4.25 bpw

E2M1 decoding (the 16 possible 4-bit values):
    0 0000  =  +0.0
    1 0001  =  +0.5      (subnormal)
    2 0010  =  +1.0
    3 0011  =  +1.5
    4 0100  =  +2.0
    5 0101  =  +3.0
    6 0110  =  +4.0
    7 0111  =  +6.0
    8 1000  =  -0.0
    9 1001  =  -0.5
   10 1010  =  -1.0
   11 1011  =  -1.5
   12 1100  =  -2.0
   13 1101  =  -3.0
   14 1110  =  -4.0
   15 1111  =  -6.0

E8M0 scale decoding: value = 2^(byte - 127)   for byte in [0, 254],  NaN at 255.

Packed layout (as observed in gpt-oss-20b safetensors):
    blocks:  shape [..., N_blocks, 16] uint8      (16 bytes per block = 32 FP4 values)
    scales:  shape [..., N_blocks]     uint8      (one E8M0 exponent per block)

The 32 FP4 values in a block are packed as follows (verified against reference):
    each byte contains 2 FP4 values: low nibble = value at 2*i, high nibble = value at 2*i+1
"""
from __future__ import annotations
import torch


# FP4 E2M1 lookup table: index = 4-bit value, output = float32.
_E2M1_LUT = torch.tensor([
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def mxfp4_dequant(blocks: torch.Tensor, scales: torch.Tensor,
                  block_size: int = 32,
                  low_nibble_first: bool = True) -> torch.Tensor:
    """
    blocks: [..., N_blocks, block_size/2] uint8  (packed 2 FP4 per byte)
    scales: [..., N_blocks]               uint8  (E8M0 exponents)
    returns: [..., N_blocks * block_size] float32

    Total un-packed size = block_size elements per block × N_blocks blocks.
    """
    assert blocks.dtype == torch.uint8
    assert scales.dtype == torch.uint8
    assert block_size % 2 == 0
    packed_per_block = block_size // 2
    assert blocks.shape[-1] == packed_per_block, \
        f"expected last dim {packed_per_block}, got {blocks.shape[-1]}"
    assert scales.shape == blocks.shape[:-1], \
        f"scales {scales.shape} vs blocks {blocks.shape[:-1]}"

    lut = _E2M1_LUT.to(blocks.device)

    # Unpack: each byte -> 2 FP4 values.
    # low nibble  (bits 0..3) is at index 2*i
    # high nibble (bits 4..7) is at index 2*i + 1
    lo = blocks & 0x0F
    hi = (blocks >> 4) & 0x0F

    if low_nibble_first:
        # interleave: [lo0, hi0, lo1, hi1, ...]
        stacked = torch.stack([lo, hi], dim=-1)
    else:
        stacked = torch.stack([hi, lo], dim=-1)
    # shape: [..., N_blocks, packed_per_block, 2]
    indices = stacked.reshape(*blocks.shape[:-1], block_size).long()

    # Decode via LUT.
    values = lut[indices]     # [..., N_blocks, block_size]

    # Apply E8M0 scale: 2^(scale - 127). scale=255 encodes NaN; we clamp to 0.
    scale_exp = scales.to(torch.int32) - 127          # [..., N_blocks]
    # 2^k via ldexp — exact powers of 2.
    scale_f = torch.ldexp(torch.ones_like(scale_exp, dtype=torch.float32),
                          scale_exp)                   # [..., N_blocks]
    # NaN encoding: scale byte 255.
    scale_f = torch.where(scales == 255,
                          torch.tensor(float('nan')),
                          scale_f)

    values = values * scale_f.unsqueeze(-1)            # [..., N_blocks, block_size]

    # Flatten the (N_blocks, block_size) tail into one dim.
    return values.reshape(*values.shape[:-2], -1)


def mxfp4_dequant_weight(blocks: torch.Tensor, scales: torch.Tensor,
                          out_features: int, in_features: int,
                          block_size: int = 32) -> torch.Tensor:
    """
    Convenience: dequant a [expert, out, N_blocks, 16] uint8 tensor into
    [expert, out, in_features] float32.
    """
    assert in_features % block_size == 0
    n_blocks = in_features // block_size
    assert blocks.shape[-2:] == (n_blocks, block_size // 2), \
        f"expected [..., {n_blocks}, {block_size//2}], got {blocks.shape[-2:]}"
    return mxfp4_dequant(blocks, scales, block_size=block_size)


if __name__ == "__main__":
    # Basic sanity: build a synthetic block and dequant it.
    # Say scale=1.0 (byte 127), and every value = +1.0 (nibble 0010=0x2).
    # So bytes = 0x22 pattern = two values of 1.0 each.
    blocks = torch.full((1, 16), 0x22, dtype=torch.uint8)
    scales = torch.tensor([127], dtype=torch.uint8)
    out = mxfp4_dequant(blocks, scales)
    print(f"synthetic block, all-1.0, scale=1.0:")
    print(f"  shape = {list(out.shape)}")
    print(f"  values = {out.flatten().tolist()}")
    assert torch.all(out == 1.0), "sanity check failed"
    print("  OK (all values equal 1.0)")
