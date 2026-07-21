import os
from typing import Callable, Optional, Union

import torch
from diffusers import HunyuanVideoPipeline
from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput

from xfuser.config import EngineConfig
from xfuser.core.distributed import get_data_parallel_rank, get_data_parallel_world_size, get_runtime_state
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
            "Diffusers HunyuanVideoPipeline is not registered with xFuser transformer parallel wrappers in this "
            f"xFuser installation. Disable unsupported modes: {', '.join(enabled)}. Data parallelism is supported."
        )


def _slice(value, start: int, end: int):
    if isinstance(value, list):
        return value[start:end]
    return value


def _slice_data_parallel(prompt, prompt_2=None, negative_prompt=None, negative_prompt_2=None):
    dp_world_size = get_data_parallel_world_size()
    if dp_world_size <= 1:
        return prompt, prompt_2, negative_prompt, negative_prompt_2

    batch_size = len(prompt) if isinstance(prompt, list) else 1
    dp_rank = get_data_parallel_rank()
    dp_batch_size = (batch_size + dp_world_size - 1) // dp_world_size
    start = dp_rank * dp_batch_size
    end = min(start + dp_batch_size, batch_size)
    if not isinstance(prompt, list) and dp_rank > 0:
        return [], [], [], []
    return (
        _slice(prompt, start, end),
        _slice(prompt_2, start, end),
        _slice(negative_prompt, start, end),
        _slice(negative_prompt_2, start, end),
    )


@xFuserPipelineWrapperRegister.register(HunyuanVideoPipeline)
class xFuserHunyuanVideoPipeline(xFuserPipelineBaseWrapper):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        _validate_supported_parallelism(engine_config)
        pipeline = HunyuanVideoPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if return_org_pipeline:
            return pipeline
        return cls(pipeline=pipeline, engine_config=engine_config)

    def __init__(self, pipeline: HunyuanVideoPipeline, engine_config: EngineConfig):
        _validate_supported_parallelism(engine_config)
        super().__init__(pipeline=pipeline, engine_config=engine_config)

    def prepare_run(self, input_config, steps: int = 1, sync_steps: int = 1):
        prompt = [""] * input_config.batch_size if input_config.batch_size > 1 else ""
        self(
            prompt=prompt,
            height=input_config.height,
            width=input_config.width,
            num_frames=input_config.num_frames,
            num_inference_steps=steps,
            guidance_scale=input_config.guidance_scale,
            true_cfg_scale=1.0,
            output_type="latent",
            generator=torch.Generator(device="cuda").manual_seed(42),
        )

    @property
    def guidance_scale(self):
        return self.module.guidance_scale

    @property
    def num_timesteps(self):
        return self.module.num_timesteps

    @property
    def attention_kwargs(self):
        return self.module.attention_kwargs

    @property
    def current_timestep(self):
        return self.module.current_timestep

    @property
    def interrupt(self):
        return self.module.interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, list[str]] = None,
        prompt_2: Optional[Union[str, list[str]]] = None,
        negative_prompt: Union[str, list[str]] = None,
        negative_prompt_2: Optional[Union[str, list[str]]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        sigmas: list[float] = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[dict] = None,
        clip_skip: Optional[int] = None,
        prompt_template: dict = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 256,
    ):
        prompt, prompt_2, negative_prompt, negative_prompt_2 = _slice_data_parallel(
            prompt, prompt_2, negative_prompt, negative_prompt_2
        )
        if isinstance(prompt, list) and not prompt:
            if not return_dict:
                return ([],)
            return HunyuanVideoPipelineOutput(frames=[])

        batch_size = len(prompt) if isinstance(prompt, list) else 1
        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            max_condition_sequence_length=max_sequence_length,
        )

        return self.module(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            clip_skip=clip_skip,
            prompt_template=prompt_template,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
