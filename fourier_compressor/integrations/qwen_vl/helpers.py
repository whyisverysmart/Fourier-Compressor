import torch

from fourier_compressor import compress


def _new_side(side_half: int, ratio: float) -> int:
    """Compute the new half-grid side after compression, keeping it >= 1."""
    return max(int(side_half * ratio), 1)


def compress_one_image(
    features: torch.Tensor,
    grid_thw: torch.Tensor,
    ratio: float = 2 / 3,
    norm: str | None = "ortho",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Compress one image's visual tokens.

    Args:
        features: tensor of shape ``[T * (H/2) * (W/2), C]`` - the post-vision-tower
            embeddings for a single image or video.
        grid_thw: 1D tensor of length 3 holding ``(T, H, W)`` BEFORE patch merge.
        ratio: fraction of frequencies to keep along each spatial axis.
        norm: ``"ortho"`` (default) or ``None``.

    Returns:
        ``(compressed_features, (new_half_h, new_half_w))`` where
        ``compressed_features`` has shape ``[T * new_half_h * new_half_w, C]``.
    """
    T = int(grid_thw[0].item())
    half_h = int(grid_thw[1].item()) // 2
    half_w = int(grid_thw[2].item()) // 2
    reserve_h = _new_side(half_h, ratio)
    reserve_w = _new_side(half_w, ratio)

    seq_len, C = features.shape
    expected = T * half_h * half_w
    if seq_len != expected:
        raise ValueError(
            f"features seq_len={seq_len} != T*(H/2)*(W/2)={expected} "
            f"(grid_thw={grid_thw.tolist()})"
        )

    x = features.view(T, half_h * half_w, C)
    x = compress(x, half_h, half_w, reserve_h, reserve_w, norm=norm)
    out = x.reshape(T * reserve_h * reserve_w, C)
    return out, (reserve_h, reserve_w)


def compress_visual_output(
    embeds: torch.Tensor,
    grid_thw: torch.Tensor,
    ratio: float = 2 / 3,
    norm: str | None = "ortho",
    update_grid_thw_inplace: bool = True,
) -> torch.Tensor:
    """Compress a batched visual-tower output and optionally update ``grid_thw`` in place.

    Args:
        embeds: tensor of shape ``[sum_i T_i*(H_i/2)*(W_i/2), C]`` - the
            concatenated visual tokens for ``len(grid_thw)`` images/videos.
        grid_thw: tensor of shape ``[num_visuals, 3]`` with each row ``(T, H, W)``
            (before Qwen's 2x merge).
        ratio: fraction kept along each spatial axis.
        update_grid_thw_inplace: if True, mutate ``grid_thw`` so downstream code
            (token replacement, RoPE) sees the new shapes. This matches Qwen's
            convention of carrying H, W in patch-pre-merge units, so we write
            back ``new_half * 2``.

    Returns:
        Concatenated compressed embeddings.
    """
    per_lens = [
        int(g[0].item() * (g[1].item() // 2) * (g[2].item() // 2)) for g in grid_thw
    ]
    chunks = torch.split(embeds, per_lens, dim=0)

    out_chunks = []
    new_sides: list[tuple[int, int]] = []
    for chunk, gt in zip(chunks, grid_thw):
        compressed, (new_h, new_w) = compress_one_image(chunk, gt, ratio=ratio, norm=norm)
        out_chunks.append(compressed)
        new_sides.append((new_h, new_w))

    if update_grid_thw_inplace:
        for i, (new_h, new_w) in enumerate(new_sides):
            grid_thw[i, 1] = max(new_h * 2, 2)
            grid_thw[i, 2] = max(new_w * 2, 2)

    out = torch.cat(out_chunks, dim=0)
    return out
