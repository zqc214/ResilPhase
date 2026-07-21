from pathlib import Path

import torch
from diffusers.utils import export_to_video
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    get_world_group,
)

from resilphase_hunyuanvideo_xdit import apply_resilphase, reset_resilphase_cache, xFuserHunyuanVideoPipeline


def _dtype_from_arg(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    parser = FlexibleArgumentParser(description="ResilPhase HunyuanVideo xDiT/xFuser inference")
    parser = xFuserArgs.add_cli_args(parser)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--fresh-threshold", type=int, default=6)
    parser.add_argument("--max-order", type=int, default=1)
    parser.add_argument("--first-enhance", type=int, default=3)
    parser.add_argument("--mapping-method", choices=["balanced", "chebyshev"], default="balanced")
    parser.add_argument("--balance-alpha", type=float, default=0.55)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = _dtype_from_arg(args.dtype)

    local_rank = get_world_group().local_rank
    pipe = xFuserHunyuanVideoPipeline.from_pretrained(
        engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=engine_config.runtime_config.dtype,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    apply_resilphase(
        pipe,
        num_steps=input_config.num_inference_steps,
        fresh_threshold=args.fresh_threshold,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        mapping_method=args.mapping_method,
        balance_alpha=args.balance_alpha,
    )

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    pipe.prepare_run(input_config, steps=1)
    reset_resilphase_cache(pipe)
    torch.cuda.reset_peak_memory_stats()

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    output = pipe(
        prompt=input_config.prompt,
        negative_prompt=input_config.negative_prompt,
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        num_inference_steps=input_config.num_inference_steps,
        guidance_scale=input_config.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        output_type=input_config.output_type,
        max_sequence_length=input_config.max_sequence_length,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time.record()
    torch.cuda.synchronize()

    elapsed_time = start_time.elapsed_time(end_time) * 1e-3
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dp_rank = get_data_parallel_rank()
    dp_world_size = get_data_parallel_world_size()
    batch_size = len(input_config.prompt) if isinstance(input_config.prompt, list) else 1
    dp_batch_size = (batch_size + dp_world_size - 1) // dp_world_size

    if input_config.output_type == "latent":
        output_path = output_dir / f"resilphase_hunyuanvideo_xdit_rank{dp_rank}.pt"
        torch.save(output.frames, output_path)
        print(f"latent output saved to {output_path}")
    else:
        for video_index, frames in enumerate(output.frames):
            global_index = dp_rank * dp_batch_size + video_index
            output_path = output_dir / f"resilphase_hunyuanvideo_xdit_rank{dp_rank}_{global_index}.mp4"
            export_to_video(frames, output_path.as_posix(), fps=args.fps)
            print(f"video {video_index} saved to {output_path}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, "
            f"parameter memory: {parameter_peak_memory / 1e9:.2f} GB, "
            f"memory: {peak_memory / 1e9:.2f} GB"
        )

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
