import argparse
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from fourier_compressor.integrations.qwen_vl import apply_to_qwen2_vl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-2B-Instruct",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", default="Describe this image briefly.")
    parser.add_argument(
        "--ratio",
        type=float,
        default=2 / 3,
        help="DCT compression ratio per spatial axis",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=256 * 28 * 28,
        help="lower bound on visual tokens per image",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=2304 * 28 * 28,
        help="upper bound on visual tokens per image",
    )
    args = parser.parse_args()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )

    if args.ratio < 1.0:
        print(f"Applying Fourier Compressor to Qwen2-VL with ratio={args.ratio}...")
        apply_to_qwen2_vl(model, processor, ratio=args.ratio)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.query},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    print(processor.batch_decode(trimmed, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
