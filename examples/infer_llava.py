import argparse
from fourier_compressor.integrations.llava import apply_to_llava


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", default="Describe this image in detail.")
    parser.add_argument(
        "--reserve",
        type=int,
        default=12,
        help="kept side length of the low-frequency block",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    # 1) Patch LLaVA BEFORE loading the model.
    assert 0 < args.reserve < 24, "Reserve should be between 0 and 24."
    print(f"Applying Fourier Compressor to LLaVA with reserve={args.reserve}...")
    apply_to_llava(reserve=args.reserve)

    # 2) Run LLaVA's official inference entry point.
    from llava.mm_utils import get_model_name_from_path
    from llava.eval.run_llava import eval_model

    eval_args = type("Args", (), {
        "model_path": args.model,
        "model_base": None,
        "model_name": get_model_name_from_path(args.model),
        "query": args.query,
        "conv_mode": None,
        "image_file": args.image,
        "video_file": None,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
    })()
    eval_model(eval_args)


if __name__ == "__main__":
    main()
