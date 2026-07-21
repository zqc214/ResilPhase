import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import T5EncoderModel
from xfuser import xFuserArgs, xFuserFluxPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    get_world_group,
)
from xfuser.core.distributed.parallel_state import get_tensor_model_parallel_world_size

from resilphase_flux_xdit import apply_resilphase, reset_resilphase_cache


def _dtype_from_arg(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    parser = FlexibleArgumentParser(description="ResilPhase FLUX xDiT/xFuser inference")
    parser = xFuserArgs.add_cli_args(parser)
    parser.add_argument("--output-dir", default="./results")
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
    text_encoder_2 = T5EncoderModel.from_pretrained(
        engine_config.model_config.model,
        subfolder="text_encoder_2",
        torch_dtype=engine_config.runtime_config.dtype,
    )

    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize

        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=engine_config.runtime_config.dtype,
        text_encoder_2=text_encoder_2,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    if not dist.is_initialized() or get_tensor_model_parallel_world_size() == 1:
        from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxTransformer2DWrapper

        if not isinstance(pipe.transformer, xFuserFluxTransformer2DWrapper):
            pipe.transformer = xFuserFluxTransformer2DWrapper(pipe.transformer)

    apply_resilphase(
        pipe,
        num_steps=input_config.num_inference_steps,
        fresh_threshold=args.fresh_threshold,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        mapping_method=args.mapping_method,
        balance_alpha=args.balance_alpha,
    )

    joint_attention_kwargs = {}
    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    pipe.prepare_run(input_config, steps=1)
    reset_resilphase_cache(pipe)
    torch.cuda.reset_peak_memory_stats()

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=input_config.max_sequence_length,
        guidance_scale=input_config.guidance_scale,
        joint_attention_kwargs=joint_attention_kwargs,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time.record()
    torch.cuda.synchronize()

    elapsed_time = start_time.elapsed_time(end_time) * 1e-3
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if input_config.output_type == "pil" and pipe.is_dp_last_group():
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        for image_index, image in enumerate(output.images):
            image_rank = dp_group_index * dp_batch_size + image_index
            image_name = f"resilphase_flux_xdit_{parallel_info}_{image_rank}.png"
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
