from typing import Dict
import torch
import math


def get_mapping_method(cache_dic: Dict) -> str:
    return cache_dic.get("mapping_method", "balanced")


def get_balance_alpha(cache_dic: Dict) -> float:
    return float(cache_dic.get("balance_alpha", 0.55))


def compute_chebyshev_nodes(num_nodes: int) -> list:
    if num_nodes <= 0:
        return []
    if num_nodes == 1:
        return [0.0]

    nodes = [math.cos((2 * k + 1) * math.pi / (2 * num_nodes)) for k in range(num_nodes)]
    nodes.sort(reverse=True)
    return nodes


def build_step_to_node_mapping(recent_steps: list, phase_nodes_desc: list) -> Dict[int, float]:
    sorted_steps = sorted(recent_steps)
    return {
        step: phase_nodes_desc[index]
        for index, step in enumerate(sorted_steps)
    }

def get_phase_mapping_from_cache(cache_dic: Dict, current_step: int) -> float:
    """
    
    :param cache_dic: Cache dictionary containing phase axis information
    :param current_step: Current step number
    :return: 当前时间步在相位轴上的映射值
    """
    if 'phase_axis' not in cache_dic:
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

def compute_stable_barycentric_weights(phase_nodes: list) -> list:
    """
    计算稳定的重心权 - 为平衡方法优化
    使用稳定的算法避免数值溢出和下溢
    
    :param phase_nodes: 相位节点列表
    :return: 稳定的重心权重列表
    """
    n = len(phase_nodes)
    if n == 0:
        return []
    
    if n == 1:
        return [1.0]
    
    weights = []
    
    # 使用对数稳定性计算
    for j in range(n):
        log_weight = 0.0
        sign = 1
        
        for k in range(n):
            if k != j:
                diff = phase_nodes[j] - phase_nodes[k]
                if abs(diff) < 1e-12:
                    # 节点重合，权重设为0
                    log_weight = float('-inf')
                    break
                
                # 使用对数计算避免数值溢出
                log_weight -= math.log(abs(diff))
                if diff < 0:
                    sign *= -1
        
        if log_weight == float('-inf'):
            weights.append(0.0)
        else:
            # 限制指数范围防止溢出
            log_weight = max(-700, min(700, log_weight))
            weights.append(sign * math.exp(log_weight))
    
    # 归一化权重（可选，但有助于数值稳定性）
    total_abs_weight = sum(abs(w) for w in weights)
    if total_abs_weight > 1e-12:
        weights = [w / total_abs_weight * len(weights) for w in weights]
    
    return weights

def store_historical_cache(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    存储完整计算时间步的历史缓存 - 存储经过所有double/single blocks后的总变化量
    用于拉格朗日插值预测，存储个数最大为 max_order + 1
    
    :param cache_dic: Cache dictionary
    :param current: Current step information
    :param feature: single-stream最终输出相对double-stream初始拼接输入的总变化量
    """
    if 'historical_cache' not in cache_dic:
        cache_dic['historical_cache'] = {
            'steps': [],
            'features': []
        }
    
    # 添加当前完整计算的结果（总变化量）
    cache_dic['historical_cache']['steps'].append(current['step'])
    cache_dic['historical_cache']['features'].append(feature.clone())
    
    # 保持最多 max_order + 1 个历史缓存
    max_cache_size = cache_dic['max_order'] + 1
    if len(cache_dic['historical_cache']['steps']) > max_cache_size:
        cache_dic['historical_cache']['steps'] = \
            cache_dic['historical_cache']['steps'][-max_cache_size:]
        cache_dic['historical_cache']['features'] = \
            cache_dic['historical_cache']['features'][-max_cache_size:]

def get_historical_cache_for_prediction(cache_dic: Dict, current: Dict) -> tuple:
    """
    获取用于预测的历史缓存
    根据当前时间步和activated_steps确定需要使用的历史缓存
    
    :param cache_dic: Cache dictionary
    :param current: Current step information
    :return: (historical_steps, historical_features) 用于插值的历史数据
    """
    if 'historical_cache' not in cache_dic:
        return [], []
    
    if 'steps' not in cache_dic['historical_cache']:
        return [], []
    
    historical_steps = cache_dic['historical_cache']['steps']
    historical_features = cache_dic['historical_cache']['features']
    
    # 根据activated_steps筛选可用的历史缓存
    # 只使用在activated_steps中的历史数据
    activated_steps = current['activated_steps']
    valid_indices = []
    
    for i, step in enumerate(historical_steps):
        if step in activated_steps:
            valid_indices.append(i)
    
    # 获取最多 max_order + 1 个最近的有效历史缓存
    max_cache_size = cache_dic['max_order'] + 1
    if len(valid_indices) > max_cache_size:
        valid_indices = valid_indices[-max_cache_size:]
    
    filtered_steps = [historical_steps[i] for i in valid_indices]
    filtered_features = [historical_features[i] for i in valid_indices]
    
    return filtered_steps, filtered_features

def barycentric_lagrange_prediction(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    使用重心形式的拉格朗日插值预测经过所有double/single blocks后的总变化量
    
    :param cache_dic: Cache dictionary containing phase axis info and barycentric weights
    :param current: Current step information
    :return: predicted_delta 预测的combined token总变化量
    """

    # 1. 获取历史缓存数据
    historical_steps, historical_features = get_historical_cache_for_prediction(cache_dic, current)
    
    # 如果只有一个历史点，直接返回该点的特征
    if len(historical_steps) == 1:
        return historical_features[0]
    
    # 2. 获取当前时间步的相位轴映射 s_target（使用平衡映射）
    s_target = get_phase_mapping_from_cache(cache_dic, current['step'])
    
    # 3. 获取历史时间步对应的相位节点和重心权
    phase_nodes = cache_dic['phase_axis']['phase_nodes']
    barycentric_weights = cache_dic['phase_axis']['barycentric_weights']
    
    # 4. 计算重心形式拉格朗日插值
    numerator = None
    denominator = 0.0
    
    for j in range(len(historical_features)):
        s_j = phase_nodes[j]  # 第j个相位节点
        w_j = barycentric_weights[j]  # 第j个重心权
        F_j = historical_features[j]  # 第j个历史特征
        
        # 计算 λ_j = w_j / (s_target - s_j)
        denominator_j = s_target - s_j
        
        if abs(denominator_j) < 1e-12:
            # 如果 s_target 与某个节点重合，直接返回该节点的特征
            return F_j
        
        lambda_j = w_j / denominator_j
        
        # 累加分子和分母
        if numerator is None:
            numerator = lambda_j * F_j
        else:
            numerator += lambda_j * F_j
        denominator += lambda_j
    
    # 5. 计算最终预测结果
    F_predicted = numerator / denominator
    
    return F_predicted

def update_phase_axis_cache(cache_dic: Dict, current: Dict):
    """
    更新相位轴缓存 - 平衡映射方法
    使用软性限制的平衡映射计算相位节点
    
    :param cache_dic: Cache dictionary
    :param current: Current step information
    """
    activated_steps = current['activated_steps']
    max_order = cache_dic['max_order']
    
    # 对activated_steps进行去重处理，保持顺序
    unique_activated_steps = []
    seen = set()
    for step in activated_steps:
        if step not in seen:
            unique_activated_steps.append(step)
            seen.add(step)
    
    # 获取最多 max_order + 1 个最近的完整计算时间步
    recent_steps = unique_activated_steps[-(max_order + 1):] if len(unique_activated_steps) >= max_order + 1 else unique_activated_steps
    
    mapping_method = get_mapping_method(cache_dic)

    if mapping_method == "chebyshev":
        n = len(recent_steps)
        if n == 0:
            phase_nodes = []
            step_to_node_mapping = {}
        else:
            chebyshev_nodes = compute_chebyshev_nodes(n)
            step_to_node_mapping = build_step_to_node_mapping(recent_steps, chebyshev_nodes)
            phase_nodes = [step_to_node_mapping[step] for step in recent_steps]

        t_mean = None
        max_distance = None
    else:
        t_mean = sum(recent_steps) / len(recent_steps)
        intervals = [step - t_mean for step in recent_steps]
        max_distance = max(abs(interval) for interval in intervals) if intervals else 0

        if max_distance > 0:
            linear_nodes = [interval / max_distance for interval in intervals]
            phase_nodes = [math.tanh(get_balance_alpha(cache_dic) * node) for node in linear_nodes]
        else:
            phase_nodes = [0.0] * len(intervals)
        step_to_node_mapping = {}
    
    # 计算稳定的重心权
    barycentric_weights = compute_stable_barycentric_weights(phase_nodes)
    
    # 存储到cache中供后续使用
    if 'phase_axis' not in cache_dic:
        cache_dic['phase_axis'] = {}
    
    cache_dic['phase_axis']['t_mean'] = t_mean
    cache_dic['phase_axis']['max_distance'] = max_distance
    cache_dic['phase_axis']['recent_steps'] = recent_steps
    cache_dic['phase_axis']['phase_nodes'] = phase_nodes
    cache_dic['phase_axis']['barycentric_weights'] = barycentric_weights
    cache_dic['phase_axis']['step_to_node_mapping'] = step_to_node_mapping

def update_lagrange_system_cache(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    更新拉格朗日插值系统的完整缓存 - 平衡映射版本
    在完整计算时间步调用，同时更新相位轴、重心权和历史特征缓存
    
    :param cache_dic: Cache dictionary
    :param current: Current step information  
    :param feature: single-stream最终输出相对double-stream初始拼接输入的总变化量
    """
    # 1. 存储历史缓存
    store_historical_cache(cache_dic, current, feature)
    
    # 2. 更新相位轴和重心权缓存（使用平衡映射）
    update_phase_axis_cache(cache_dic, current)

def lagrange_cache_init(cache_dic: Dict, current: Dict):
    """
    初始化拉格朗日插值缓存
    
    :param cache_dic: Cache dictionary
    :param current: Current step information
    """
    # 初始化历史缓存存储
    if 'historical_cache' not in cache_dic:
        cache_dic['historical_cache'] = {
            'steps': [],
            'features': []
        }
