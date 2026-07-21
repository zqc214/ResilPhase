def cal_type(cache_dic: dict, current: dict) -> None:
    first_steps = current["step"] > (current["num_steps"] - cache_dic["first_enhance"] - 1)
    fresh_interval = cache_dic["interval"]

    if first_steps or cache_dic["cache_counter"] == fresh_interval - 1:
        current["type"] = "full"
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])
    else:
        cache_dic["cache_counter"] += 1
        current["type"] = "ResilPhase"
