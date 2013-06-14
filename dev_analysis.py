'''
Created on Jun 12, 2013
Workspace for developing AoMR sampler results analyses
@author: goker
'''
import cPickle as pickle
from mcmc_sampler import *
from vision_forward_model import VisionForwardModel

# load results
fname = 'AoMR_Shape_Grammar_-_Visual_Condition20130613201442.mcmc'
f = open(fname)
results = pickle.load(f)
print results
state = results.best_samples[0][0].state
forward_model = VisionForwardModel()
forward_model._view(state)

state.tree.show()

