'''
Created on Jun 12, 2013
Workspace for developing AoMR sampler results analyses
@author: goker
'''
import cPickle as pickle
from mcmc_sampler import *
from vision_forward_model import VisionForwardModel

# load results
fname = './results/AoMRShapeGrammar_Visual_Obj120130620205934.mcmc'
f = open(fname)
results = pickle.load(f)
# print results
best1 = results.best_samples[0][0].state
print(best1)
best1.tree.show()

# load results
fname = './results/AoMRShapeGrammar_Visual_Obj220130624001257.mcmc'
f = open(fname)
results = pickle.load(f)
# print results
best2 = results.best_samples[0][0].state
print(best2)
best2.tree.show()

# load results
fname = './results/AoMRShapeGrammar_Visual_Obj1020130708020033.mcmc'
f = open(fname)
results = pickle.load(f)
# print results
best3 = results.best_samples[0][0].state
print(best3)
best3.tree.show()

print([(best1.tree[node].identifier, str(best1.tree[node].tag), best1.spatial_model.voxels.get(node)) for node in best1.tree.nodes])
print(best1.spatial_model.voxels)
print(best2.spatial_model.voxels)
print(best3.spatial_model.voxels)

print(best1.kernel_subtree(best2))
print
print(best1.kernel_subtree(best3))
print
print(best2.kernel_subtree(best3))
# 
# forward_model = VisionForwardModel()
# i = 0
# for sample in results.best_samples[0]:
#     sample.state.tree.show()
#     forward_model._view(sample.state)
#     i += 1
# 
# print('\n------------------------------------\nSamples')
# i = 0
# for sample in results.samples[0]:
#     forward_model._view(sample.state)
#     sample.state.tree.show()
#     i += 1

