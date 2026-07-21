import argparse
from pathlib import Path

import torch
from diffusers import FluxPipeline

from resilphase_diffusers import apply_resilphase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--prompt", default="An image of a squirrel in Picasso style")
    parser.add_argument("--output", default="resilphase_flux_diffusers.png")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fresh-threshold", type=int, default=6)
    parser.add_argument("--max-order", type=int, default=1)
    parser.add_argument("--first-enhance", type=int, default=3)
    parser.add_argument("--mapping-method", choices=["balanced", "chebyshev"], default="balanced")
    parser.add_argument("--balance-alpha", type=float, default=0.55)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.to("cuda")

    apply_resilphase(
        pipe,
        num_steps=args.steps,
        fresh_threshold=args.fresh_threshold,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        mapping_method=args.mapping_method,
        balance_alpha=args.balance_alpha,
    )

    image = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
