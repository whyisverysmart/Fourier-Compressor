from .helpers import compress_visual_output


def apply_to_qwen2_vl(
    model,
    processor=None,
    ratio: float = 2 / 3,
    norm: str | None = "ortho",
):
    """Patch ``model.visual.forward`` AND ``processor.__call__`` to apply DCT compression.

    Args:
        model: a ``Qwen2VLForConditionalGeneration`` instance (anything with a
            ``.visual`` submodule whose ``forward`` takes ``(pixel_values, grid_thw)``).
        processor: the matching ``Qwen2VLProcessor`` instance. **Required** for
            correct behavior — without it the processor still emits the
            uncompressed placeholder count and the model's forward will
            assert-fail. We accept ``None`` only so users who patch the
            processor manually can opt out.
        ratio: fraction of frequencies to keep per spatial axis (default 2/3).
        norm: ``"ortho"`` (default) or ``None``.

    Returns:
        The same ``model`` (for chaining).
    """
    _patch_visual(model, ratio=ratio, norm=norm)

    if processor is not None:
        from .processor_patch import patch_processor
        patch_processor(processor, ratio=ratio)
    else:
        import warnings
        warnings.warn(
            "apply_to_qwen2_vl was called without a processor; you MUST patch "
            "the processor separately (see fourier_compressor.integrations."
            "qwen2_vl.processor_patch) or use the source-edit option, otherwise "
            "the forward will fail with an n_image_tokens != n_image_features "
            "assertion.",
            stacklevel=2,
        )
    return model


def _patch_visual(model, ratio: float, norm: str | None):
    visual = model.visual
    if getattr(visual, "_fourier_compressor_patched", False):
        visual._fourier_compressor_ratio = ratio
        visual._fourier_compressor_norm = norm
        return

    original_forward = visual.forward

    def patched_forward(pixel_values, grid_thw):
        embeds = original_forward(pixel_values, grid_thw=grid_thw)
        current_ratio = visual._fourier_compressor_ratio
        current_norm = visual._fourier_compressor_norm
        out = compress_visual_output(
            embeds, grid_thw, ratio=current_ratio, norm=current_norm,
            update_grid_thw_inplace=True,
        )
        return out

    visual.forward = patched_forward
    visual._fourier_compressor_patched = True
    visual._fourier_compressor_ratio = ratio
    visual._fourier_compressor_norm = norm
