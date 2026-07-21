import argparse
from pathlib import Path

import torch
from diffusers import DiTPipeline

from resilphase_dit_diffusers import apply_resilphase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/DiT-XL-2-256")
    parser.add_argument("--class-label", type=int, default=495)
    parser.add_argument("--output", default="resilphase_dit_diffusers.png")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--first-enhance", type=int, default=2)
    parser.add_argument("--mapping-method", choices=["chebyshev", "balanced"], default="chebyshev")
    parser.add_argument("--balance-alpha", type=float, default=0.55)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.to("cuda")

    apply_resilphase(
        pipe,
        num_steps=args.steps,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        mapping_method=args.mapping_method,
        balance_alpha=args.balance_alpha,
    )

    image = pipe(
        class_labels=[args.class_label],
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
