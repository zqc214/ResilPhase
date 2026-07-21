import math
from typing import Dict

import torch


def get_mapping_method(cache_dic: Dict) -> str:
    return cache_dic.get("mapping_method", "balanced")


def get_balance_alpha(cache_dic: Dict) -> float:
    return float(cache_dic.get("balance_alpha", 0.55))


def compute_chebyshev_nodes(num_nodes: int) -> list[float]:
    if num_nodes <= 0:
        return []
    if num_nodes == 1:
        return [0.0]

    nodes = [math.cos((2 * k + 1) * math.pi / (2 * num_nodes)) for k in range(num_nodes)]
    nodes.sort(reverse=True)
    return nodes


def build_step_to_node_mapping(recent_steps: list[int], phase_nodes_desc: list[float]) -> dict[int, float]:
    sorted_steps = sorted(recent_steps)
    return {step: phase_nodes_desc[index] for index, step in enumerate(sorted_steps)}


def get_phase_mapping_from_cache(cache_dic: Dict, current_step: int) -> float:
    if "phase_axis" not in cache_dic:
        return 0.0

    phase_info = cache_dic["phase_axis"]
    mapping_method = get_mapping_method(cache_dic)

    if mapping_method == "chebyshev":
        recent_steps = phase_info.get("recent_steps", [])
        step_to_node_mapping = phase_info.get("step_to_node_mapping", {})

        if current_step in step_to_node_mapping:
            return step_to_node_mapping[current_step]
        if not recent_steps:
            return 0.0
        if len(recent_steps) == 1:
            return 0.0

        recent_steps_sorted = sorted(recent_steps)
        if current_step <= recent_steps_sorted[0]:
            step1, step2 = recent_steps_sorted[0], recent_steps_sorted[1]
            node1, node2 = step_to_node_mapping[step1], step_to_node_mapping[step2]
            slope = (node2 - node1) / (step2 - step1)
            return max(-1.0, node1 + slope * (current_step - step1))

        if current_step >= recent_steps_sorted[-1]:
            step1, step2 = recent_steps_sorted[-2], recent_steps_sorted[-1]
            node1, node2 = step_to_node_mapping[step1], step_to_node_mapping[step2]
            slope = (node2 - node1) / (step2 - step1)
            return min(1.0, node2 + slope * (current_step - step2))

        left_step = max(step for step in recent_steps_sorted if step <= current_step)
        right_step = min(step for step in recent_steps_sorted if step >= current_step)
        if left_step == right_step:
            return step_to_node_mapping[left_step]

        left_node = step_to_node_mapping[left_step]
        right_node = step_to_node_mapping[right_step]
        ratio = (current_step - left_step) / (right_step - left_step)
        return left_node + ratio * (right_node - left_node)

    t_mean = phase_info.get("t_mean", 0)
    max_distance = phase_info.get("max_distance", 1)
    if max_distance == 0:
        return 0.0

    current_interval = current_step - t_mean
    linear_mapping = current_interval / max_distance
    return math.tanh(get_balance_alpha(cache_dic) * linear_mapping)


def compute_stable_barycentric_weights(phase_nodes: list[float]) -> list[float]:
    n = len(phase_nodes)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    weights = []
    for j in range(n):
        log_weight = 0.0
        sign = 1

        for k in range(n):
            if k == j:
                continue
            diff = phase_nodes[j] - phase_nodes[k]
            if abs(diff) < 1e-12:
                log_weight = float("-inf")
                break

            log_weight -= math.log(abs(diff))
            if diff < 0:
                sign *= -1

        if log_weight == float("-inf"):
            weights.append(0.0)
        else:
            log_weight = max(-700, min(700, log_weight))
            weights.append(sign * math.exp(log_weight))

    total_abs_weight = sum(abs(weight) for weight in weights)
    if total_abs_weight > 1e-12:
        weights = [weight / total_abs_weight * len(weights) for weight in weights]

    return weights


def store_historical_cache(
    cache_dic: Dict,
    current: Dict,
    img_delta_double: torch.Tensor,
    txt_delta_double: torch.Tensor,
    img_delta_single: torch.Tensor,
) -> None:
    if "historical_cache" not in cache_dic:
        cache_dic["historical_cache"] = {
            "steps": [],
            "img_double": [],
            "txt_double": [],
            "img_single": [],
        }

    cache_dic["historical_cache"]["steps"].append(current["step"])
    cache_dic["historical_cache"]["img_double"].append(img_delta_double.detach().clone())
    cache_dic["historical_cache"]["txt_double"].append(txt_delta_double.detach().clone())
    cache_dic["historical_cache"]["img_single"].append(img_delta_single.detach().clone())

    max_cache_size = cache_dic["max_order"] + 1
    if len(cache_dic["historical_cache"]["steps"]) > max_cache_size:
        for key in ("steps", "img_double", "txt_double", "img_single"):
            cache_dic["historical_cache"][key] = cache_dic["historical_cache"][key][-max_cache_size:]


def get_historical_cache_for_prediction(cache_dic: Dict, current: Dict) -> tuple:
    historical_cache = cache_dic.get("historical_cache")
    if not historical_cache or "steps" not in historical_cache:
        return [], [], [], []

    historical_steps = historical_cache["steps"]
    activated_steps = current["activated_steps"]
    valid_indices = [index for index, step in enumerate(historical_steps) if step in activated_steps]

    max_cache_size = cache_dic["max_order"] + 1
    if len(valid_indices) > max_cache_size:
        valid_indices = valid_indices[-max_cache_size:]

    return (
        [historical_cache["steps"][index] for index in valid_indices],
        [historical_cache["img_double"][index] for index in valid_indices],
        [historical_cache["txt_double"][index] for index in valid_indices],
        [historical_cache["img_single"][index] for index in valid_indices],
    )


def barycentric_lagrange_prediction_double(cache_dic: Dict, current: Dict) -> tuple[torch.Tensor, torch.Tensor]:
    historical_steps, img_double_features, txt_double_features, _ = get_historical_cache_for_prediction(
        cache_dic, current
    )
    if len(historical_steps) == 0:
        raise RuntimeError("ResilPhase cache has no historical double-block features.")
    if len(historical_steps) == 1:
        return img_double_features[0], txt_double_features[0]

    s_target = get_phase_mapping_from_cache(cache_dic, current["step"])
    phase_nodes = cache_dic["phase_axis"]["phase_nodes"]
    barycentric_weights = cache_dic["phase_axis"]["barycentric_weights"]

    img_numerator = None
    txt_numerator = None
    denominator = 0.0

    for j, (img_feature, txt_feature) in enumerate(zip(img_double_features, txt_double_features)):
        denominator_j = s_target - phase_nodes[j]
        if abs(denominator_j) < 1e-12:
            return img_feature, txt_feature

        lambda_j = barycentric_weights[j] / denominator_j
        img_numerator = lambda_j * img_feature if img_numerator is None else img_numerator + lambda_j * img_feature
        txt_numerator = lambda_j * txt_feature if txt_numerator is None else txt_numerator + lambda_j * txt_feature
        denominator += lambda_j

    return img_numerator / denominator, txt_numerator / denominator


def barycentric_lagrange_prediction_single(cache_dic: Dict, current: Dict) -> torch.Tensor:
    historical_steps, _, _, img_single_features = get_historical_cache_for_prediction(cache_dic, current)
    if len(historical_steps) == 0:
        raise RuntimeError("ResilPhase cache has no historical single-block features.")
    if len(historical_steps) == 1:
        return img_single_features[0]

    s_target = get_phase_mapping_from_cache(cache_dic, current["step"])
    phase_nodes = cache_dic["phase_axis"]["phase_nodes"]
    barycentric_weights = cache_dic["phase_axis"]["barycentric_weights"]

    numerator = None
    denominator = 0.0

    for j, feature in enumerate(img_single_features):
        denominator_j = s_target - phase_nodes[j]
        if abs(denominator_j) < 1e-12:
            return feature

        lambda_j = barycentric_weights[j] / denominator_j
        numerator = lambda_j * feature if numerator is None else numerator + lambda_j * feature
        denominator += lambda_j

    return numerator / denominator


def update_phase_axis_cache(cache_dic: Dict, current: Dict) -> None:
    unique_activated_steps = []
    seen = set()
    for step in current["activated_steps"]:
        if step not in seen:
            unique_activated_steps.append(step)
            seen.add(step)

    max_order = cache_dic["max_order"]
    recent_steps = unique_activated_steps[-(max_order + 1) :]
    mapping_method = get_mapping_method(cache_dic)

    if mapping_method == "chebyshev":
        chebyshev_nodes = compute_chebyshev_nodes(len(recent_steps))
        step_to_node_mapping = build_step_to_node_mapping(recent_steps, chebyshev_nodes)
        phase_nodes = [step_to_node_mapping[step] for step in recent_steps]
        t_mean = None
        max_distance = None
    elif recent_steps:
        t_mean = sum(recent_steps) / len(recent_steps)
        intervals = [step - t_mean for step in recent_steps]
        max_distance = max(abs(interval) for interval in intervals) if intervals else 0
        if max_distance > 0:
            linear_nodes = [interval / max_distance for interval in intervals]
            phase_nodes = [math.tanh(get_balance_alpha(cache_dic) * node) for node in linear_nodes]
        else:
            phase_nodes = [0.0] * len(intervals)
        step_to_node_mapping = {}
    else:
        t_mean = 0.0
        max_distance = 0.0
        phase_nodes = []
        step_to_node_mapping = {}

    cache_dic["phase_axis"] = {
        "t_mean": t_mean,
        "max_distance": max_distance,
        "recent_steps": recent_steps,
        "phase_nodes": phase_nodes,
        "barycentric_weights": compute_stable_barycentric_weights(phase_nodes),
        "step_to_node_mapping": step_to_node_mapping,
    }


def update_lagrange_system_cache(
    cache_dic: Dict,
    current: Dict,
    img_delta_double: torch.Tensor,
    txt_delta_double: torch.Tensor,
    img_delta_single: torch.Tensor,
) -> None:
    store_historical_cache(cache_dic, current, img_delta_double, txt_delta_double, img_delta_single)
    update_phase_axis_cache(cache_dic, current)
