from .config import ResilPhaseDiTConfig


def init_cache(config: ResilPhaseDiTConfig, num_layers: int) -> tuple[dict, dict]:
    cache = {-1: {}}
    for layer_index in range(num_layers):
        cache[-1][layer_index] = {}

    cache_dic = {
        "cache": cache,
        "interval": config.interval,
        "max_order": config.max_order,
        "mapping_method": config.mapping_method,
        "balance_alpha": config.balance_alpha,
        "first_enhance": config.first_enhance,
        "cache_counter": 0,
        "cal_threshold": config.interval,
    }
    current = {
        "num_steps": config.num_steps,
        "activated_steps": [],
        "call_index": 0,
        "step": config.num_steps - 1,
        "type": "full",
    }
    return cache_dic, current
