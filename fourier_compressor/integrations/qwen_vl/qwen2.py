from .monkey_patch import apply_to_qwen_vl


def apply_to_qwen2_vl(
    model,
    processor=None,
    ratio: float = 2 / 3,
    norm: str | None = "ortho",
):
    """Patch a Qwen2-VL model and processor with DCT visual-token compression."""
    return apply_to_qwen_vl(
        model,
        processor=processor,
        ratio=ratio,
        norm=norm,
        model_name="Qwen2-VL",
    )
