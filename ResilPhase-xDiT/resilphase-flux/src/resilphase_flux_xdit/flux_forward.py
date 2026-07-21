from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

from .cache import init_cache
from .interpolation import (
    barycentric_lagrange_prediction_double,
    barycentric_lagrange_prediction_single,
    update_lagrange_system_cache,
)
from .scheduler import cal_type

try:
    from xfuser.core.distributed.parallel_state import is_pipeline_first_stage
except Exception:
    is_pipeline_first_stage = None

logger = logging.get_logger(__name__)


def _pipeline_first_stage() -> bool:
    return is_pipeline_first_stage is None or is_pipeline_first_stage()


def _after_single_blocks(self) -> bool:
    stage_info = getattr(self, "stage_info", None)
    if stage_info is None:
        return True
    return stage_info.after_flags.get("single_transformer_blocks", True)


def _uses_xfuser_stage_output(self) -> bool:
    return getattr(self, "stage_info", None) is not None


def _get_resilphase_state(self: FluxTransformer2DModel) -> tuple[dict, dict]:
    config = self._resilphase_config
    cache_dic = getattr(self, "_resilphase_cache_dic", None)
    current = getattr(self, "_resilphase_current", None)

    if cache_dic is None or current is None or current.get("step", 0) >= config.num_steps:
        cache_dic, current = init_cache(
            config=config,
            num_layers=self.config.num_layers,
            num_single_layers=self.config.num_single_layers,
        )
        self._resilphase_cache_dic = cache_dic
        self._resilphase_current = current

    return cache_dic, current


def _maybe_checkpoint(self, fn, *args):
    if torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False):
        if hasattr(self, "_gradient_checkpointing_func"):
            return self._gradient_checkpointing_func(fn, *args)

        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        return torch.utils.checkpoint.checkpoint(fn, *args, **ckpt_kwargs)

    return fn(*args)


def _run_double_block(self, block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, attention_kwargs):
    def call_with_kwargs(hidden, encoder, emb, rotary):
        try:
            return block(
                hidden_states=hidden,
                encoder_hidden_states=encoder,
                temb=emb,
                image_rotary_emb=rotary,
                joint_attention_kwargs=attention_kwargs,
            )
        except TypeError as exc:
            if "joint_attention_kwargs" not in str(exc):
                raise
            return block(
                hidden_states=hidden,
                encoder_hidden_states=encoder,
                temb=emb,
                image_rotary_emb=rotary,
            )

    return _maybe_checkpoint(self, call_with_kwargs, hidden_states, encoder_hidden_states, temb, image_rotary_emb)


def _run_single_block(self, block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, attention_kwargs):
    def call_new_signature(hidden, encoder, emb, rotary):
        try:
            return block(
                hidden_states=hidden,
                encoder_hidden_states=encoder,
                temb=emb,
                image_rotary_emb=rotary,
                joint_attention_kwargs=attention_kwargs,
            )
        except TypeError as exc:
            if "encoder_hidden_states" not in str(exc) and "joint_attention_kwargs" not in str(exc):
                raise

            combined = torch.cat([encoder, hidden], dim=1)
            try:
                combined = block(
                    hidden_states=combined,
                    temb=emb,
                    image_rotary_emb=rotary,
                    joint_attention_kwargs=attention_kwargs,
                )
            except TypeError as old_exc:
                if "joint_attention_kwargs" not in str(old_exc):
                    raise
                combined = block(
                    hidden_states=combined,
                    temb=emb,
                    image_rotary_emb=rotary,
                )
            return combined[:, : encoder.shape[1], ...], combined[:, encoder.shape[1] :, ...]

    result = _maybe_checkpoint(self, call_new_signature, hidden_states, encoder_hidden_states, temb, image_rotary_emb)
    if isinstance(result, tuple) and len(result) == 2:
        return result

    combined = result
    return combined[:, : encoder_hidden_states.shape[1], ...], combined[:, encoder_hidden_states.shape[1] :, ...]


def resilphase_xfuser_flux_forward(
    self: FluxTransformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    if controlnet_block_samples is not None or controlnet_single_block_samples is not None:
        original_forward = getattr(self, "_resilphase_original_forward", None)
        if original_forward is None:
            raise NotImplementedError("ResilPhase ControlNet fallback requires the original forward method.")
        return original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=return_dict,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )

    cache_dic, current = _get_resilphase_state(self)
    cal_type(cache_dic, current)

    if timestep is not None:
        current["t"] = float(timestep.flatten()[0].detach().cpu())

    attention_kwargs = joint_attention_kwargs.copy() if joint_attention_kwargs is not None else {}
    lora_scale = attention_kwargs.pop("scale", 1.0)

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
        logger.warning("Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.")

    if _pipeline_first_stage():
        hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )

    if _pipeline_first_stage():
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated. "
            "Please remove the batch dimension and pass it as a 2d torch Tensor."
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated. "
            "Please remove the batch dimension and pass it as a 2d torch Tensor."
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if "ip_adapter_image_embeds" in attention_kwargs:
        ip_adapter_image_embeds = attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    if current["type"] == "full":
        img_input = hidden_states.clone()
        txt_input = encoder_hidden_states.clone()

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = _run_double_block(
                self,
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                attention_kwargs,
            )

        img_delta_double = hidden_states - img_input
        txt_delta_double = encoder_hidden_states - txt_input

        text_seq_len = encoder_hidden_states.shape[1]
        single_input = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = _run_single_block(
                self,
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                attention_kwargs,
            )

        single_output = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        img_delta_single = single_output - single_input

        update_lagrange_system_cache(
            cache_dic=cache_dic,
            current=current,
            img_delta_double=img_delta_double,
            txt_delta_double=txt_delta_double,
            img_delta_single=img_delta_single,
        )

        hidden_states = single_output[:, text_seq_len:]
        encoder_hidden_states = single_output[:, :text_seq_len]

    elif current["type"] == "resilphase_cache":
        img_delta_double_pred, txt_delta_double_pred = barycentric_lagrange_prediction_double(cache_dic, current)
        hidden_states = hidden_states + img_delta_double_pred
        encoder_hidden_states = encoder_hidden_states + txt_delta_double_pred

        text_seq_len = encoder_hidden_states.shape[1]
        single_input = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        single_output = single_input + barycentric_lagrange_prediction_single(cache_dic, current)
        encoder_hidden_states = single_output[:, :text_seq_len]
        hidden_states = single_output[:, text_seq_len:]

    else:
        raise NotImplementedError(f"Unsupported ResilPhase calculation type: {current['type']}")

    if _after_single_blocks(self):
        hidden_states = self.norm_out(hidden_states, temb)
        output = (self.proj_out(hidden_states), None) if _uses_xfuser_stage_output(self) else self.proj_out(hidden_states)
    else:
        output = hidden_states, encoder_hidden_states

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    current["step"] += 1

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
