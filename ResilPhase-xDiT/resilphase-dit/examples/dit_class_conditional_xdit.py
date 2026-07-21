import argparse
from pathlib import Path

import torch
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import get_data_parallel_rank, get_data_parallel_world_size, get_runtime_state, get_world_group

from resilphase_dit_xdit import apply_resilphase, reset_resilphase_cache, xFuserDiTPipeline


def _dtype_from_arg(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _parse_class_labels(value: str) -> list[int]:
    labels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("--class-labels must contain at least one integer label.")
    return labels


def main() -> None:
    parser = FlexibleArgumentParser(description="ResilPhase DiT xDiT/xFuser inference")
    parser = xFuserArgs.add_cli_args(parser)
    parser.add_argument("--class-labels", type=_parse_class_labels, default=[495])
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--first-enhance", type=int, default=2)
    parser.add_argument("--mapping-method", choices=["chebyshev", "balanced"], default="chebyshev")
    parser.add_argument("--balance-alpha", type=float, default=0.55)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = _dtype_from_arg(args.dtype)

    local_rank = get_world_group().local_rank
    pipe = xFuserDiTPipeline.from_pretrained(
        engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=engine_config.runtime_config.dtype,
    )
    pipe = pipe.to(f"cuda:{local_rank}")

    apply_resilphase(
        pipe.module,
        num_steps=input_config.num_inference_steps,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        mapping_method=args.mapping_method,
        balance_alpha=args.balance_alpha,
    )

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    pipe.prepare_run(input_config, steps=1)
    reset_resilphase_cache(pipe.module)
    torch.cuda.reset_peak_memory_stats()

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    output = pipe(
        class_labels=args.class_labels,
        guidance_scale=input_config.guidance_scale,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
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
    dp_batch_size = (len(args.class_labels) + dp_world_size - 1) // dp_world_size
    for image_index, image in enumerate(output.images):
        global_index = dp_rank * dp_batch_size + image_index
        image_name = f"resilphase_dit_xdit_dp{dp_world_size}_rank{dp_rank}_{global_index}.png"
        image.save(output_dir / image_name)
        print(f"image {image_index} saved to {output_dir / image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, "
            f"parameter memory: {parameter_peak_memory / 1e9:.2f} GB, "
            f"memory: {peak_memory / 1e9:.2f} GB"
        )

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
