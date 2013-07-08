'''
Created on Jun 12, 2013
Workspace for developing AoMR sampler results analyses
@author: goker
'''
import cPickle as pickle
from mcmc_sampler import *
from vision_forward_model import VisionForwardModel

# load results
fname = 'AoMRShapeGrammar_Visual_Obj820130624004057.mcmc'
f = open(fname)
results = pickle.load(f)
print results

print(results.best_samples[0][0].state.ll_params)

forward_model = VisionForwardModel()
i = 0
for sample in results.best_samples[0]:
    sample.state.tree.show()
    #forward_model._view(sample.state)
    forward_model.save_image('best_sample'+str(i)+'.png', sample.state)
    i += 1

print('\n------------------------------------\nSamples')
i = 0
for sample in results.samples[0]:
    #forward_model._view(sample.state)
    sample.state.tree.show()
    forward_model.save_image('sample'+str(i)+'.png', sample.state)
    i += 1

