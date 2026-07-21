def force_scheduler(cache_dic: dict, current: dict) -> None:
    linear_step_weight = 0.0
    step_factor = 1 - linear_step_weight + 2 * linear_step_weight * current["step"] / current["num_steps"]
    cache_dic["cal_threshold"] = max(1, round(cache_dic["fresh_threshold"] / step_factor))


def cal_type(cache_dic: dict, current: dict) -> None:
    if cache_dic["fresh_ratio"] == 0.0 and not cache_dic["resilphase_cache"]:
        first_step = current["step"] == 0
    else:
        first_step = current["step"] < cache_dic["first_enhance"]

    fresh_interval = cache_dic["fresh_threshold"] if first_step else cache_dic["cal_threshold"]

    if first_step or cache_dic["cache_counter"] == fresh_interval - 1:
        current["type"] = "full"
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])
        force_scheduler(cache_dic, current)
    elif cache_dic["resilphase_cache"]:
        cache_dic["cache_counter"] += 1
        current["type"] = "resilphase_cache"
    elif cache_dic["Delta-DiT"]:
        cache_dic["cache_counter"] += 1
        current["type"] = "Delta-Cache"
    else:
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
