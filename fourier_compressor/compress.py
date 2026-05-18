import math
import torch

from .dct import dct2, idct2


def compress(
    features: torch.Tensor,
    grid_h: int,
    grid_w: int,
    reserve_h: int,
    reserve_w: int,
    norm: str | None = "ortho",
) -> torch.Tensor:
    """Compress a sequence of visual tokens by keeping a low-frequency DCT block.

    Args:
        features: tensor of shape ``[..., grid_h * grid_w, C]``, **fp16 or bf16**.
        grid_h, grid_w: spatial layout of the input tokens.
        reserve_h, reserve_w: side lengths of the kept low-frequency block.
            The output sequence has ``reserve_h * reserve_w`` tokens.
        norm: ``"ortho"`` (default) or ``None``; passed through to
            :func:`fourier_compressor.dct.dct2`.

    Returns:
        Tensor of shape ``[..., reserve_h * reserve_w, C]`` with the same dtype
        as the input.
    """
    *batch, N, C = features.shape
    if N != grid_h * grid_w:
        raise ValueError(
            f"features second-to-last dim ({N}) does not match grid_h*grid_w "
            f"({grid_h}*{grid_w}={grid_h * grid_w})"
        )
    if reserve_h > grid_h or reserve_w > grid_w:
        raise ValueError(
            f"reserve ({reserve_h}, {reserve_w}) must be <= grid ({grid_h}, {grid_w})"
        )

    orig_dtype = features.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"compress() expects float16 or bfloat16 input, got {orig_dtype}. "
            "Cast your features to fp16/bf16 before calling compress()."
        )

    # [..., N, C] -> [..., C, H, W], cast to fp32 for the DCT only.
    x = features.float().transpose(-1, -2).reshape(*batch, C, grid_h, grid_w)
    X = dct2(x, norm=norm)
    X_crop = X[..., :reserve_h, :reserve_w]
    x_back = idct2(X_crop, norm=norm)
    out = x_back.reshape(*batch, C, reserve_h * reserve_w).transpose(-1, -2)
    return out.to(orig_dtype)


def compress_square(
    features: torch.Tensor,
    reserve: int = 12,
    norm: str | None = "ortho",
) -> torch.Tensor:
    """Convenience wrapper for square grids (e.g. LLaVA's 24x24 CLIP tokens)."""
    *_, N, _ = features.shape
    grid = int(math.isqrt(N))
    if grid * grid != N:
        raise ValueError(
            f"features second-to-last dim ({N}) is not a perfect square; "
            "pass `grid_h` and `grid_w` to compress() instead."
        )
    return compress(features, grid, grid, reserve, reserve, norm=norm)
