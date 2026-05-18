from fourier_compressor import compress_square


def apply_to_llava(reserve: int = 12, norm: str | None = "ortho"):
    """Monkey-patch ``LlavaMetaForCausalLM.encode_images`` to apply DCT compression.

    Args:
        reserve: side length of the kept low-frequency block; output sequence
            has ``reserve * reserve`` tokens (e.g. ``reserve=12`` -> 144 tokens,
            a 4x reduction from LLaVA's 576).
        norm: ``"ortho"`` (default) or ``None``.

    Returns:
        The patched ``LlavaMetaForCausalLM`` class.
    """
    try:
        from llava.model.llava_arch import LlavaMetaForCausalLM
    except ImportError as e:
        raise ImportError(
            "LLaVA is not importable. Install it from "
            "https://github.com/haotian-liu/LLaVA before calling apply_to_llava()."
        ) from e

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = compress_square(image_features, reserve=reserve, norm=norm)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    LlavaMetaForCausalLM.encode_images = encode_images
    LlavaMetaForCausalLM._fourier_compressor_patched = True
    return LlavaMetaForCausalLM
