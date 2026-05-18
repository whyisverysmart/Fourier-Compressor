# LLaVA Integration

Two ways to use Fourier Compressor with LLaVA.

## Option A — Monkey-patch (recommended)

No source modification. Add **one line** before loading the model:

```python
from fourier_compressor.integrations.llava import apply_to_llava

apply_to_llava(reserve=12)   # 576 -> 144 visual tokens

# ... then load LLaVA as usual:
from llava.model.builder import load_pretrained_model
tokenizer, model, image_processor, _ = load_pretrained_model(...)
```

`reserve=12` gives a 4x reduction (24x24 -> 12x12). Adjust as you like.

## Option B — Source edit

Edit `llava/model/llava_arch.py` directly:

```diff
 from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, ...
 from llava.mm_utils import get_anyres_image_grid_shape
+from fourier_compressor import compress_square

 ...
     def encode_images(self, images):
         image_features = self.get_model().get_vision_tower()(images)
+        image_features = compress_square(image_features, reserve=12)
         image_features = self.get_model().mm_projector(image_features)
         return image_features
```

Then `pip install -e .` LLaVA as usual.
