import os
from typing import Optional, Union

import torch
from diffusers import DiTPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from xfuser.config import EngineConfig
from xfuser.core.distributed import get_data_parallel_rank, get_data_parallel_world_size
from xfuser.model_executor.pipelines import xFuserPipelineBaseWrapper
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister


def _validate_supported_parallelism(engine_config: EngineConfig) -> None:
    parallel_config = engine_config.parallel_config
    unsupported = {
        "sequence parallel": parallel_config.sp_degree,
        "tensor parallel": parallel_config.tp_degree,
        "pipeline parallel": parallel_config.pp_degree,
        "CFG parallel": parallel_config.cfg_degree,
        "parallel VAE": parallel_config.vae_parallel_size,
    }
    enabled = [name for name, degree in unsupported.items() if degree > 1]
    if enabled:
        raise RuntimeError(
            "Diffusers DiTPipeline is not registered with xFuser transformer parallel wrappers in this xFuser "
            f"installation. Disable unsupported modes: {', '.join(enabled)}. Data parallelism is supported."
        )


def _slice_data_parallel(class_labels: list[int]) -> list[int]:
    dp_world_size = get_data_parallel_world_size()
    if dp_world_size <= 1:
        return class_labels

    dp_rank = get_data_parallel_rank()
    batch_size = len(class_labels)
    dp_batch_size = (batch_size + dp_world_size - 1) // dp_world_size
    start = dp_rank * dp_batch_size
    end = min(start + dp_batch_size, batch_size)
    return class_labels[start:end]


@xFuserPipelineWrapperRegister.register(DiTPipeline)
class xFuserDiTPipeline(xFuserPipelineBaseWrapper):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        _validate_supported_parallelism(engine_config)
        pipeline = DiTPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if return_org_pipeline:
            return pipeline
        return cls(pipeline=pipeline, engine_config=engine_config)

    def __init__(self, pipeline: DiTPipeline, engine_config: EngineConfig):
        _validate_supported_parallelism(engine_config)
        super().__init__(pipeline=pipeline, engine_config=engine_config)

    def prepare_run(self, input_config, steps: int = 1, sync_steps: int = 1):
        class_labels = [0] * max(1, input_config.batch_size)
        self(
            class_labels=class_labels,
            guidance_scale=input_config.guidance_scale,
            num_inference_steps=steps,
            generator=torch.Generator(device="cuda").manual_seed(42),
            output_type=input_config.output_type,
        )

    @torch.no_grad()
    def __call__(
        self,
        class_labels: list[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        class_labels = _slice_data_parallel(class_labels)
        if not class_labels:
            return ImagePipelineOutput(images=[])

        return self.module(
            class_labels=class_labels,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            return_dict=return_dict,
        )
