from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.models import DiTTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .cache import init_cache
from .interpolation import barycentric_lagrange_prediction, lagrange_cache_init, update_lagrange_system_cache
from .scheduler import cal_type


def _get_resilphase_state(self: DiTTransformer2DModel) -> tuple[dict, dict]:
    config = self._resilphase_config
    cache_dic = getattr(self, "_resilphase_cache_dic", None)
    current = getattr(self, "_resilphase_current", None)

    if cache_dic is None or current is None or current.get("call_index", 0) >= config.num_steps:
        cache_dic, current = init_cache(config=config, num_layers=self.config.num_layers)
        self._resilphase_cache_dic = cache_dic
        self._resilphase_current = current

    return cache_dic, current


def _run_block(self, block, hidden_states, timestep, cross_attention_kwargs, class_labels):
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        return self._gradient_checkpointing_func(
            block,
            hidden_states,
            None,
            None,
            None,
            timestep,
            cross_attention_kwargs,
            class_labels,
        )

    return block(
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=timestep,
        cross_attention_kwargs=cross_attention_kwargs,
        class_labels=class_labels,
    )


def resilphase_dit_forward(
    self: DiTTransformer2DModel,
    hidden_states: torch.Tensor,
    timestep: Optional[torch.LongTensor] = None,
    class_labels: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    return_dict: bool = True,
):
    cache_dic, current = _get_resilphase_state(self)
    current["step"] = current["num_steps"] - 1 - current["call_index"]
    if timestep is not None:
        current["t"] = float(timestep.flatten()[0].detach().cpu())

    height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
    hidden_states = self.pos_embed(hidden_states)

    cal_type(cache_dic, current)
    if current["type"] == "ResilPhase" and "historical_cache" not in cache_dic:
        current["type"] = "full"
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])

    if current["type"] == "full":
        hidden_states_before_blocks = hidden_states.clone()
        lagrange_cache_init(cache_dic)

        for block in self.transformer_blocks:
            hidden_states = _run_block(self, block, hidden_states, timestep, cross_attention_kwargs, class_labels)

        update_lagrange_system_cache(cache_dic, current, hidden_states - hidden_states_before_blocks)

    elif current["type"] == "ResilPhase":
        hidden_states = hidden_states + barycentric_lagrange_prediction(cache_dic, current)

    else:
        raise NotImplementedError(f"Unsupported ResilPhase calculation type: {current['type']}")

    conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=hidden_states.dtype)
    shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
    hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
    hidden_states = self.proj_out_2(hidden_states)

    height = width = int(hidden_states.shape[1] ** 0.5)
    hidden_states = hidden_states.reshape(
        shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
    )

    current["call_index"] += 1

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
