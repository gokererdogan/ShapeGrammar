'''
Created on Jun 14, 2013
MCMC Sampler script for AoMR Shape Grammar
@author: goker
'''

import numpy as np
from aomr_grammar import AoMRShapeState, AoMRSpatialModel
from vision_forward_model import VisionForwardModel
from mcmc_sampler import MCMCSampler
import sys
from haptics_forward_model import HapticsForwardModel

if __name__ == '__main__':
    # AoMR Simple Shape Grammar, visual condition
    obj_id = int(sys.argv[1])
    b = float(sys.argv[2])
    forward_model_type = sys.argv[3] #v, h or vh
    spatial_model = AoMRSpatialModel()
    if forward_model_type == 'v':
        forward_model = VisionForwardModel(body_fixed=False)
        npy_path = 'data/visual'
        info = 'AoMRShapeGrammar Visual Obj{0:d}'.format(obj_id)
    elif forward_model_type == 'h':
        forward_model = HapticsForwardModel(body_fixed=False)
        npy_path = 'data/haptic'
        info = 'AoMRShapeGrammar Haptic Obj{0:d}'.format(obj_id)
    elif forward_model_type == 'vh':
        raise NotImplementedError()
    else:
        raise Exception('Forward model type can be v, h or vh')
    
    data = np.load('{0:s}/{1:d}.npy'.format(npy_path, obj_id))
    
    # visual condition
    # b parameter for objects
    # obj1: 1200
    # obj2: 1600
    # obj3: 1500
    # obj4: 1600
    # obj5: 1300
    # obj6: 1500
    # obj7: 1200
    # obj8: 1200
    # obj9: 1200
    # obj10: 1500
    # obj11: 1200
    # obj12: 1500
    # obj13: 1200
    # obj14: 1400
    # obj15: 1200
    # obj16: 1200
    # haptic condition
    # b parameter for object
    # obj1: 6000
    
    state_params = {'b': b}
    init_state = AoMRShapeState(forward_model=forward_model, data=data, 
                                ll_params=state_params, spatial_model=spatial_model)
    sampler_params ={'info' : info, 
                 'runs' : 1,
                 'iters' : 10000, 
                 'keep_top_n' : 20, 
                 'burn_in' : 0,
                 'thinning_period' : 400,
                 'random_move' : False,
                 'results_folder' : './results/',
                 'save_results' : True,
                 'verbose': True}
    ms = MCMCSampler(sampler_params, init_state)
    results = ms.run()
    print results
        
    #forward_model._view(results.best_samples[0][0].state)
