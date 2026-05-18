# Qwen2-VL Integration

Qwen2-VL needs **two** synchronized changes, not one:

1. **`Qwen2VLForConditionalGeneration.forward`** — compress the visual-tower output from `T*(H/2)*(W/2)` tokens down to `T*new_h*new_w` tokens.
2. **`Qwen2VLProcessor.__call__`** — emit `T*new_h*new_w` `<|image_pad|>` placeholder tokens per image (instead of the original `T*(H/2)*(W/2)`).

If only (1) is applied, the model's

```python
if n_image_tokens != n_image_features:
    raise ValueError(...)
```

assertion fires immediately on the first forward pass. Both patches must agree on `new_h` / `new_w`.

## Option A — Monkey-patch (recommended)

`apply_to_qwen2_vl(model, processor, ratio)` patches both at once:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from fourier_compressor.integrations.qwen2_vl import apply_to_qwen2_vl

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
apply_to_qwen2_vl(model, processor, ratio=2/3)
```

Pass the processor — without it, `apply_to_qwen2_vl` will warn and the forward will assert-fail on the first image. Handles video inputs without further changes (Qwen reuses the same visual tower for both; our patch splits per-image via `grid_thw`).

Tested against `transformers==4.51.3`, which is the version used for the paper release.

## Option B — Source edit

Two files to edit. Both changes use the **same** integer formula so the post-compression count is consistent on both sides.

### 1. `transformers/models/qwen2_vl/modeling_qwen2_vl.py`

```diff
+from fourier_compressor.integrations.qwen2_vl import compress_visual_output

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

`compress_visual_output` mutates `image_grid_thw` / `video_grid_thw` in place so the downstream token-replacement and M-RoPE code sees the new shapes.

### 2. `transformers/models/qwen2_vl/processing_qwen2_vl.py`

Inside `Qwen2VLProcessor.__call__`, change the per-image placeholder count:

```diff
     if image_grid_thw is not None:
-        merge_length = self.image_processor.merge_size**2
         index = 0
         for i in range(len(text)):
             while self.image_token in text[i]:
                 text[i] = text[i].replace(
                     self.image_token,
-                    "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
+                    "<|placeholder|>" * (
+                        int(image_grid_thw[index][0])
+                        * max(int(image_grid_thw[index][1]) // 2 * 2 // 3, 1)
+                        * max(int(image_grid_thw[index][2]) // 2 * 2 // 3, 1)
+                    ),
                     1,
                 )
                 index += 1
             text[i] = text[i].replace("<|placeholder|>", self.image_token)
```

...and the same edit for the `video_grid_thw` branch.

For convenience, the same count is exposed as a helper so you don't duplicate the formula:

```python
from fourier_compressor.integrations.qwen2_vl import compressed_placeholder_count
count = compressed_placeholder_count(grid_thw, ratio=2/3)
```
