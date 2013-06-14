'''
Created on Jun 14, 2013
MCMC sampler script for Rational Rules model
@author: goker
'''

import numpy as np
from rational_rules import RationalRulesState
from mcmc_sampler import MCMCSampler, MCMCRunResults

if __name__ == '__main__':
    # Rational Rules model learning
    data = np.array([[0, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1], [0, 0, 1, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0]])
    state_params = {'b': 6.0}
    init_state = RationalRulesState(data=data, ll_params=state_params)
    sampler_params ={'info' : 'Rational Rules', 
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
    print(results)

