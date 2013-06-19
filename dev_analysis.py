'''
Created on Jun 12, 2013
Workspace for developing AoMR sampler results analyses
@author: goker
'''
import cPickle as pickle
from mcmc_sampler import *
from vision_forward_model import VisionForwardModel

# load results
fname = 'AoMR_Shape_Grammar_-_Visual_Condition20130617135710.mcmc'
f = open(fname)
results = pickle.load(f)
print results
forward_model = VisionForwardModel()
for sample in results.samples[0]:
    forward_model._view(sample.state)
    sample.state.tree.show()

