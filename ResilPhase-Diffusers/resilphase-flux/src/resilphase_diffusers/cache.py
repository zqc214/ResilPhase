from .config import ResilPhaseConfig


def init_cache(config: ResilPhaseConfig, num_layers: int, num_single_layers: int) -> tuple[dict, dict]:
    cache: dict = {-1: {"double_stream": {}, "single_stream": {}}}
    cache_index: dict = {-1: {}, "layer_index": {}}
    attn_map: dict = {-1: {"double_stream": {}, "single_stream": {}}}

    for index in range(num_layers):
        cache[-1]["double_stream"][index] = {}
        cache_index[-1][index] = {}
        attn_map[-1]["double_stream"][index] = {
            "total": {},
            "txt_mlp": {},
            "img_mlp": {},
        }

    for index in range(num_single_layers):
        cache[-1]["single_stream"][index] = {}
        cache_index[-1][index] = {}
        attn_map[-1]["single_stream"][index] = {"total": {}}

    cache_dic = {
        "attn_map": attn_map,
        "cache": cache,
        "cache_index": cache_index,
        "cache_counter": 0,
        "cache_type": "random",
        "fresh_ratio_schedule": "ToCa",
        "fresh_ratio": 0.0,
        "fresh_threshold": config.fresh_threshold,
        "force_fresh": "global",
        "soft_fresh_weight": 0.0,
        "resilphase_cache": True,
        "Delta-DiT": False,
        "max_order": config.max_order,
        "first_enhance": config.first_enhance,
        "mapping_method": config.mapping_method,
        "balance_alpha": config.balance_alpha,
        "cal_threshold": config.fresh_threshold,
    }
    current = {
        "activated_steps": [0],
        "step": 0,
        "num_steps": config.num_steps,
        "type": "full",
    }
    return cache_dic, current
