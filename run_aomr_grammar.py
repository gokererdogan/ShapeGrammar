'''
Created on Jun 14, 2013
MCMC Sampler script for AoMR Shape Grammar
@author: goker
'''

import numpy as np
from aomr_grammar import AoMRShapeState, AoMRSpatialModel
from vision_forward_model import VisionForwardModel
from mcmc_sampler import MCMCSampler

if __name__ == '__main__':
    # AoMR Simple Shape Grammar, visual condition
    # successfully learned objects
    # 1: AoMR_Shape_Grammar_-_Visual_Condition20130618102128
    # 3: AoMR_Shape_Grammar_-_Visual_Condition20130618142453
    
    spatial_model = AoMRSpatialModel()
    forward_model = VisionForwardModel()
    data = np.load('data/visual/1.npy')
    # b parameter for objects
    # obj1: 1600
    # obj2: 2000
    # obj3: 2600
    # obj16: 2400
    state_params = {'b': 1200.0}
    init_state = AoMRShapeState(forward_model=forward_model, data=data, 
                                ll_params=state_params, spatial_model=spatial_model)
    sampler_params ={'info' : 'AoMRShapeGrammar Visual Obj1', 
                 'runs' : 1,
                 'iters' : 10000, 
                 'keep_top_n' : 20, 
                 'burn_in' : 0,
                 'thinning_period' : 400,
                 'random_move' : False,
                 'results_folder' : './',
                 'save_results' : True,
                 'verbose': True}
    ms = MCMCSampler(sampler_params, init_state)
    results = ms.run()
    print results
        
    forward_model._view(results.best_samples[0][0].state)
