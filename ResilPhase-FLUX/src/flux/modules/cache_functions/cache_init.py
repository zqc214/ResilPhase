def cache_init(timesteps, model_kwargs=None):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    cache_dic['attn_map'][-1]['double_stream'] = {}
    cache_dic['attn_map'][-1]['single_stream'] = {}

    cache_dic['k-norm'] = {}
    cache_dic['k-norm'][-1] = {}
    cache_dic['k-norm'][-1]['double_stream'] = {}
    cache_dic['k-norm'][-1]['single_stream'] = {}

    cache_dic['v-norm'] = {}
    cache_dic['v-norm'][-1] = {}
    cache_dic['v-norm'][-1]['double_stream'] = {}
    cache_dic['v-norm'][-1]['single_stream'] = {}

    cache_dic['cross_attn_map'] = {}
    cache_dic['cross_attn_map'][-1] = {}
    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache_dic['cache_counter'] = 0

    for j in range(19):
        cache[-1]['double_stream'][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1]['double_stream'][j] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['total'] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['txt_mlp'] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['img_mlp'] = {}
        
        cache_dic['k-norm'][-1]['double_stream'][j] = {}
        cache_dic['k-norm'][-1]['double_stream'][j]['txt_mlp'] = {}
        cache_dic['k-norm'][-1]['double_stream'][j]['img_mlp'] = {}

        cache_dic['v-norm'][-1]['double_stream'][j] = {}
        cache_dic['v-norm'][-1]['double_stream'][j]['txt_mlp'] = {}
        cache_dic['v-norm'][-1]['double_stream'][j]['img_mlp'] = {}

    for j in range(38):
        cache[-1]['single_stream'][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j]['total'] = {}

        cache_dic['k-norm'][-1]['single_stream'][j] = {}
        cache_dic['k-norm'][-1]['single_stream'][j]['total'] = {}

        cache_dic['v-norm'][-1]['single_stream'][j] = {}
        cache_dic['v-norm'][-1]['single_stream'][j]['total'] = {}

    cache_dic['resilphase_cache'] = False
    cache_dic['Delta-DiT'] = False

    mode = 'ResilPhase'

    if mode == 'original':
        cache_dic['cache_type'] = 'random' 
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa'
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 1
        cache_dic['force_fresh'] = 'global'
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 3
        
    elif mode == 'ToCa':
        cache_dic['cache_type'] = 'attention'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.1
        cache_dic['fresh_threshold'] = 6
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 3
    
    elif mode == 'ResilPhase':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 6
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['resilphase_cache'] = True
        cache_dic['max_order'] = 1
        cache_dic['first_enhance'] = 3
        # Phase-axis mapping configuration for ResilPhase.
        # mapping_method: 'balanced' or 'chebyshev'
        # balance_alpha is only used when mapping_method == 'balanced'
        cache_dic['mapping_method'] = 'balanced'
        cache_dic['balance_alpha'] = 0.55

    elif mode == 'Delta':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 3
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['Delta-DiT'] = True
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 1

    current = {}
    current['final_time'] = timesteps[-2]
    current['activated_steps'] = [0]

    return cache_dic, current
