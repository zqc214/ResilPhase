from types import MethodType
from typing import Any

from diffusers.models import HunyuanVideoTransformer3DModel

from .config import ResilPhaseHunyuanVideoConfig
from .hunyuanvideo_forward import resilphase_hunyuanvideo_forward


def _coerce_config(
    config: ResilPhaseHunyuanVideoConfig | None,
    kwargs: dict[str, Any],
) -> ResilPhaseHunyuanVideoConfig:
    if config is not None and kwargs:
        raise ValueError("Pass either config or keyword options, not both.")
    if config is None:
        config = ResilPhaseHunyuanVideoConfig(**kwargs)
    config.validate()
    return config


def _is_hunyuanvideo_transformer_like(transformer) -> bool:
    if isinstance(transformer, HunyuanVideoTransformer3DModel):
        return True
    return hasattr(transformer, "transformer_blocks") and hasattr(transformer, "single_transformer_blocks")


def apply_resilphase(pipe_or_transformer, config: ResilPhaseHunyuanVideoConfig | None = None, **kwargs):
    config = _coerce_config(config, kwargs)
    transformer = getattr(pipe_or_transformer, "transformer", pipe_or_transformer)
    if not _is_hunyuanvideo_transformer_like(transformer):
        raise TypeError("apply_resilphase expects a HunyuanVideo pipeline or HunyuanVideo transformer.")

    if not hasattr(transformer, "_resilphase_original_forward"):
        transformer._resilphase_original_forward = transformer.forward

    transformer._resilphase_config = config
    transformer._resilphase_cache_dic = None
    transformer._resilphase_current = None
    transformer.forward = MethodType(resilphase_hunyuanvideo_forward, transformer)
    return pipe_or_transformer


def remove_resilphase(pipe_or_transformer):
    transformer = getattr(pipe_or_transformer, "transformer", pipe_or_transformer)
    original_forward = getattr(transformer, "_resilphase_original_forward", None)
    if original_forward is not None:
        transformer.forward = original_forward
        delattr(transformer, "_resilphase_original_forward")

    for attr in ("_resilphase_config", "_resilphase_cache_dic", "_resilphase_current"):
        if hasattr(transformer, attr):
            delattr(transformer, attr)
    return pipe_or_transformer


def reset_resilphase_cache(pipe_or_transformer) -> None:
    transformer = getattr(pipe_or_transformer, "transformer", pipe_or_transformer)
    if hasattr(transformer, "_resilphase_cache_dic"):
        transformer._resilphase_cache_dic = None
    if hasattr(transformer, "_resilphase_current"):
        transformer._resilphase_current = None
