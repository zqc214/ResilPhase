from typing import Any, Dict, Optional, Union

import torch
from diffusers.models import HunyuanVideoTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from .cache import init_cache
from .interpolation import (
    barycentric_lagrange_prediction_double,
    barycentric_lagrange_prediction_single,
    update_lagrange_system_cache,
)
from .scheduler import cal_type

logger = logging.get_logger(__name__)


def _get_resilphase_state(self: HunyuanVideoTransformer3DModel) -> tuple[dict, dict]:
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


def _run_block(self, block, hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb,
               token_replace_emb, first_frame_num_tokens):
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        return self._gradient_checkpointing_func(
            block,
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_mask,
            image_rotary_emb,
            token_replace_emb,
            first_frame_num_tokens,
        )

    return block(
        hidden_states,
        encoder_hidden_states,
        temb,
        attention_mask,
        image_rotary_emb,
        token_replace_emb,
        first_frame_num_tokens,
    )


def resilphase_hunyuanvideo_forward(
    self: HunyuanVideoTransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
        logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    cache_dic, current = _get_resilphase_state(self)
    cal_type(cache_dic, current)
    if timestep is not None:
        current["t"] = float(timestep.flatten()[0].detach().cpu())

    batch_size, _, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = post_patch_height * post_patch_width

    image_rotary_emb = self.rope(hidden_states)
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.ones(batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool)
    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
    indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
    mask_indices = indices >= effective_sequence_length.unsqueeze(1)
    attention_mask = attention_mask.masked_fill(mask_indices, False)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    if current["type"] == "full":
        hidden_input = hidden_states.clone()
        encoder_input = encoder_hidden_states.clone()

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = _run_block(
                self,
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

        img_delta_double = hidden_states - hidden_input
        txt_delta_double = encoder_hidden_states - encoder_input

        single_input = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = _run_block(
                self,
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

        single_output = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        img_delta_single = single_output - single_input

        update_lagrange_system_cache(
            cache_dic=cache_dic,
            current=current,
            img_delta_double=img_delta_double,
            txt_delta_double=txt_delta_double,
            img_delta_single=img_delta_single,
        )

        hidden_states = single_output[:, :latent_sequence_length]

    elif current["type"] == "resilphase_cache":
        img_delta_double_pred, txt_delta_double_pred = barycentric_lagrange_prediction_double(cache_dic, current)
        hidden_states = hidden_states + img_delta_double_pred
        encoder_hidden_states = encoder_hidden_states + txt_delta_double_pred

        single_input = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        single_output = single_input + barycentric_lagrange_prediction_single(cache_dic, current)
        hidden_states = single_output[:, :latent_sequence_length]

    else:
        raise NotImplementedError(f"Unsupported ResilPhase calculation type: {current['type']}")

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    current["step"] += 1

    if not return_dict:
        return (hidden_states,)

    return Transformer2DModelOutput(sample=hidden_states)
