# Qwen-VL Integration

This directory contains the shared integration for Qwen2-VL and Qwen2.5-VL. The two model families use different transformers class names, but the relevant interfaces are the same under `transformers==4.51.3`.
## Option A — Monkey-patch (recommended)

```python
from fourier_compressor.integrations.qwen_vl import apply_to_qwen2_vl

apply_to_qwen2_vl(model, processor, ratio=2/3)
# ... then load model as usual.
```

```python
from fourier_compressor.integrations.qwen_vl import apply_to_qwen2_5_vl

apply_to_qwen2_5_vl(model, processor, ratio=2/3)
# ... then load model as usual.
```

Both entry points call the same shared implementation. Pass the processor so the visual features and placeholder token counts stay synchronized.

## Option B — Source edit

Two synchronized edits are required for both Qwen2-VL and Qwen2.5-VL.

### Model Forward

Use `compress_visual_output` after the visual tower:

```diff
+from fourier_compressor.integrations.qwen_vl import compress_visual_output

     if pixel_values is not None:
         pixel_values = pixel_values.type(self.visual.get_dtype())
         image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
+        image_embeds = compress_visual_output(image_embeds, image_grid_thw, ratio=2/3)
         ...

     if pixel_values_videos is not None:
         pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
         video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
+        video_embeds = compress_visual_output(video_embeds, video_grid_thw, ratio=2/3)
         ...
```

`compress_visual_output` mutates `image_grid_thw` / `video_grid_thw` in place so downstream token replacement and M-RoPE see the compressed shapes.

### Processor Placeholder Count

Inside the processor `__call__`, replace the original placeholder count with:

```python
from fourier_compressor.integrations.qwen_vl import compressed_placeholder_count

count = compressed_placeholder_count(grid_thw, ratio=2/3)
```

Apply the same count formula to image and video branches.

Relevant transformers files:

- Qwen2-VL: `transformers/models/qwen2_vl/modeling_qwen2_vl.py`, `transformers/models/qwen2_vl/processing_qwen2_vl.py`
- Qwen2.5-VL: `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`, `transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py`
