import math
import torch


def dct(x: torch.Tensor, norm: str | None = "ortho") -> torch.Tensor:
    """DCT-II of `x` over the last dimension.

    Args:
        x: real-valued tensor of any shape; transform is taken along ``dim=-1``.
        norm: ``"ortho"`` for an orthonormal transform, ``None`` for the un-normalized
            (scipy-default) variant.

    Returns:
        Tensor of the same shape as ``x``.
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v, dim=1)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


def idct(X: torch.Tensor, norm: str | None = "ortho") -> torch.Tensor:
    """Inverse of :func:`dct` (i.e. ``idct(dct(x)) == x`` up to numerical error)."""
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, N) / 2
    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.view_as_complex(V)

    v = torch.fft.ifft(V, dim=1).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]
    return x.view(*x_shape)


def dct2(x: torch.Tensor, norm: str | None = "ortho") -> torch.Tensor:
    """2D DCT-II over the last two dimensions of ``x``.

    ``x`` may have arbitrary leading dimensions, e.g. ``[B, C, H, W]`` or
    ``[C, T, H, W]``. The transform is taken over ``(H, W)``.
    """
    X = dct(x, norm=norm)
    X = dct(X.transpose(-1, -2), norm=norm).transpose(-1, -2)
    return X


def idct2(X: torch.Tensor, norm: str | None = "ortho") -> torch.Tensor:
    """Inverse of :func:`dct2`."""
    x = idct(X.transpose(-1, -2), norm=norm).transpose(-1, -2)
    return idct(x, norm=norm)
