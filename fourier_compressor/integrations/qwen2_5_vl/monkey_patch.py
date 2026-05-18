from fourier_compressor.integrations.qwen2_vl.monkey_patch import apply_to_qwen2_vl


def apply_to_qwen2_5_vl(
    model,
    processor=None,
    ratio: float = 2 / 3,
    norm: str | None = "ortho",
):
    """Patch a Qwen2.5-VL model + processor with DCT visual-token compression.

    Args:
        model: a ``Qwen2_5_VLForConditionalGeneration`` instance.
        processor: the matching ``Qwen2_5_VLProcessor``. **Required** for
            correct behavior — see :func:`apply_to_qwen2_vl` for why.
        ratio: fraction of frequencies to keep per spatial axis (default 2/3).
        norm: ``"ortho"`` (default) or ``None``.

    Returns:
        The same ``model`` (for chaining).
    """
    return apply_to_qwen2_vl(model, processor, ratio=ratio, norm=norm)
