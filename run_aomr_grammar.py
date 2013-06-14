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
    spatial_model = AoMRSpatialModel()
    forward_model = VisionForwardModel()
    data = np.load('data/visual/1.npy')
    state_params = {'b': 1750.0}
    init_state = AoMRShapeState(forward_model=forward_model, data=data, 
                                ll_params=state_params, spatial_model=spatial_model)
    sampler_params ={'info' : 'AoMR Shape Grammar - Visual Condition', 
                 'runs' : 1,
                 'iters' : 5000, 
                 'keep_top_n' : 20, 
                 'burn_in' : 1000,
                 'thinning_period' : 400,
                 'random_move' : False,
                 'results_folder' : './',
                 'save_results' : True,
                 'verbose': True}
    ms = MCMCSampler(sampler_params, init_state)
    results = ms.run()
    print results
        
    forward_model._view(results.best_samples[0][0].state)
