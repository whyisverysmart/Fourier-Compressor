import importlib
import torch


def _new_half(half_grid: int, ratio: float) -> int:
    """Compressed half-grid side. Matches the integer formula ``(half*2)//3`` for ratio=2/3."""
    return max(int(half_grid * ratio), 1)


def compressed_placeholder_count(grid_thw, ratio: float = 2 / 3) -> int:
    """Placeholder tokens for ONE image / video after DCT compression.

    ``grid_thw`` is the pre-merge patch grid ``(T, H, W)``; it may be a 1-D
    tensor, list, or tuple. Returns ``T * new_h * new_w`` where
    ``new_h = max((H//2) * ratio, 1)`` and likewise for ``new_w``.
    """
    if isinstance(grid_thw, torch.Tensor):
        T, H, W = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    else:
        T, H, W = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    return T * _new_half(H // 2, ratio) * _new_half(W // 2, ratio)


def _get_processor_kwargs_cls(processor):
    """Auto-detect the ``<ProcessorName>Kwargs`` class for a Qwen-VL processor.

    For ``Qwen2VLProcessor`` returns ``Qwen2VLProcessorKwargs``; for
    ``Qwen2_5_VLProcessor`` returns ``Qwen2_5_VLProcessorKwargs``. Looks up
    the class in the same module as ``type(processor)``.
    """
    cls = type(processor)
    kwargs_name = cls.__name__ + "Kwargs"
    module = importlib.import_module(cls.__module__)
    try:
        return getattr(module, kwargs_name)
    except AttributeError as e:
        raise ImportError(
            f"Could not locate {kwargs_name} in {cls.__module__}. "
            f"patch_processor expects a Qwen-VL processor whose ProcessorKwargs "
            f"class follows the `<ProcessorName>Kwargs` naming convention "
            f"(tested on Qwen2-VL and Qwen2.5-VL). Fall back to the source-edit "
            f"option for unsupported versions."
        ) from e


def patch_processor(processor, ratio: float = 2 / 3):
    """Replace ``type(processor).__call__`` with a version that uses the compressed count.

    The patch is applied on the **class** of ``processor``, so it affects every
    instance of that class in the current process. Calling this twice is safe
    (idempotent); the second call only updates ``ratio``.
    """
    cls = type(processor)
    if getattr(cls, "_fourier_compressor_patched", False):
        cls._fourier_compressor_ratio = ratio
        return processor

    try:
        from transformers.feature_extraction_utils import BatchFeature
    except ImportError as e:
        raise ImportError(
            "patch_processor requires the `transformers` package."
        ) from e

    ProcessorKwargs = _get_processor_kwargs_cls(processor)

    def patched_call(self, images=None, text=None, videos=None, **kwargs):
        output_kwargs = self._merge_kwargs(
            ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(
                images=images, videos=None, **output_kwargs["images_kwargs"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            # Use videos_kwargs if present (the matching processor's ProcessorKwargs
            # typically declares it); fall back to images_kwargs for older versions.
            videos_kw = output_kwargs.get("videos_kwargs", output_kwargs["images_kwargs"])
            videos_inputs = self.image_processor(
                images=None, videos=videos, **videos_kw
            )
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        r = cls._fourier_compressor_ratio

        if image_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    count = compressed_placeholder_count(image_grid_thw[index], r)
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * count, 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    count = compressed_placeholder_count(video_grid_thw[index], r)
                    text[i] = text[i].replace(
                        self.video_token, "<|placeholder|>" * count, 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    cls.__call__ = patched_call
    cls._fourier_compressor_patched = True
    cls._fourier_compressor_ratio = ratio
    return processor
