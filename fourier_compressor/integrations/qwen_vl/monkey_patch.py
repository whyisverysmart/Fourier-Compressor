import warnings

from .helpers import compress_visual_output
from .processor_patch import patch_processor


def apply_to_qwen_vl(
    model,
    processor=None,
    ratio: float = 2 / 3,
    norm: str | None = "ortho",
    model_name: str = "Qwen-VL",
):
    """Patch a Qwen-VL-family model and processor with DCT visual-token compression.

    Qwen2-VL and Qwen2.5-VL share the relevant interfaces under the supported
    transformers versions: ``model.visual.forward(pixel_values, grid_thw)`` and
    processor-side placeholder expansion from ``grid_thw``.
    """
    _patch_visual(model, ratio=ratio, norm=norm)

    if processor is not None:
        patch_processor(processor, ratio=ratio)
    else:
        warnings.warn(
            f"{model_name} integration was called without a processor; you MUST "
            "patch the processor separately (see "
            "fourier_compressor.integrations.qwen_vl.processor_patch) or use the "
            "source-edit option, otherwise the forward will fail with an "
            "n_image_tokens != n_image_features assertion.",
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
            embeds,
            grid_thw,
            ratio=current_ratio,
            norm=current_norm,
            update_grid_thw_inplace=True,
        )
        return out

    visual.forward = patched_forward
    visual._fourier_compressor_patched = True
    visual._fourier_compressor_ratio = ratio
    visual._fourier_compressor_norm = norm
