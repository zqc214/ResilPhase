def cache_init(model_kwargs, num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache[-1]={}

    for j in range(28):
        cache[-1][j] = {}
    for i in range(num_steps):
        cache[i]={}
        for j in range(28):
            cache[i][j] = {}

    cache_dic['cache']                = cache
    cache_dic['flops']                = 0.0
    cache_dic['interval']             = model_kwargs['interval']
    cache_dic['max_order']            = model_kwargs['max_order']
    cache_dic['test_FLOPs']           = model_kwargs['test_FLOPs']
    cache_dic['mapping_method']       = model_kwargs.get('mapping_method', 'chebyshev')
    cache_dic['balance_alpha']        = float(model_kwargs.get('balance_alpha', 0.55))
    cache_dic['first_enhance']        = 2
    cache_dic['cache_counter']        = 0
    
    current = {}
    current['num_steps'] = num_steps
    # current['activated_steps'] = [49]
    current['activated_steps'] = []
    
    # 调试信息
    # print(f"[DEBUG] Cache init: steps={num_steps}, interval={cache_dic['interval']}, max_order={cache_dic['max_order']}")
    
    return cache_dic, current
    
