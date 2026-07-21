import argparse
from pathlib import Path

import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video

from resilphase_hunyuanvideo_diffusers import apply_resilphase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--prompt", default="A cat walks on the grass, realistic style.")
    parser.add_argument("--output", default="resilphase_hunyuanvideo_diffusers.mp4")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=65)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
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

    pipe = HunyuanVideoPipeline.from_pretrained(args.model, torch_dtype=dtype)
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

    frames = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).frames[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, output.as_posix(), fps=15)


if __name__ == "__main__":
    main()
