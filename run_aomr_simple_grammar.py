'''
Created on Jun 14, 2013
MCMC Sampler script for AoMR Simple Shape Grammar
@author: goker
'''

import numpy as np
from aomr_simple_grammar import AoMRSimpleShapeState, AoMRSimpleSpatialModel
from vision_forward_model import VisionForwardModel
from mcmc_sampler import MCMCSampler

if __name__ == '__main__':
    # AoMR Simple Shape Grammar, visual condition
    spatial_model = AoMRSimpleSpatialModel()
    forward_model = VisionForwardModel()
    data = np.load('data/visual/1.npy')
    state_params = {'b': 750.0}
    init_state = AoMRSimpleShapeState(forward_model, data, state_params, spatial_model)
    sampler_params ={'info' : 'aoMR Simple Grammar - Visual Condition', 
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
