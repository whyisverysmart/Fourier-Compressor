# Qwen2.5-VL Integration

Same logic as the [Qwen2-VL integration](../qwen2_vl/README.md) — Qwen2.5-VL uses different class names (`Qwen2_5_VLForConditionalGeneration`, `Qwen2_5_VLProcessor`) but the visual tower's `forward(pixel_values, grid_thw)` signature and the processor's placeholder-count logic are unchanged. `patch_processor` auto-detects the right `Qwen2_5_VLProcessorKwargs` class from the processor instance, so the underlying implementation is shared.

This release targets `transformers==4.51.3`.

## Option A — Monkey-patch (recommended)

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from fourier_compressor.integrations.qwen2_5_vl import apply_to_qwen2_5_vl

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
apply_to_qwen2_5_vl(model, processor, ratio=2/3)
```

Pass the processor — without it the model's `n_image_tokens != n_image_features` assertion will fire on the first image.

## Option B — Source edit

Identical to the Qwen2-VL source-edit instructions, with these substitutions in the file paths:

- `transformers/models/qwen2_vl/modeling_qwen2_vl.py` → `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- `transformers/models/qwen2_vl/processing_qwen2_vl.py` → `transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py`

Use the **same** `compress_visual_output` call and the **same** placeholder formula — see [the Qwen2-VL README](../qwen2_vl/README.md) for the diff template.
