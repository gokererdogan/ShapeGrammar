'''
Created on Jun 12, 2013
Workspace for developing AoMR sampler results analyses
@author: goker
'''
import cPickle as pickle
from mcmc_sampler import *
from vision_forward_model import VisionForwardModel

# load results
fname = 'AoMRShapeGrammar_Visual_Obj120130620001945.mcmc'
f = open(fname)
results = pickle.load(f)
print results
forward_model = VisionForwardModel()
for sample in results.best_samples[0]:
    sample.state.tree.show()
    forward_model._view(sample.state)

print('\n------------------------------------\nSamples')
for sample in results.samples[0]:
    forward_model._view(sample.state)
    sample.state.tree.show()

