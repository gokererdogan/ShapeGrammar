'''
Created on Jun 17, 2013
AoMRShapeGrammar class tests
@author: goker
'''
from treelib import Tree
from pcfg_tree import ParseNode
from aomr_grammar import AoMRShapeState, AoMRSpatialModel
from vision_forward_model import VisionForwardModel
import numpy as np

forward_model = VisionForwardModel()
data = np.load('data/visual/1.npy')
params = {'b': 1750.0}

# add a new branch using addbranch move

# initial state, object with bottom, front and top parts
t = Tree()
t.create_node(ParseNode('S', 2), identifier='S')
t.create_node(ParseNode('S', 6), parent='S', identifier='S1')
t.create_node(ParseNode('S', 0), parent='S', identifier='S2')
t.create_node(ParseNode('S', 0), parent='S', identifier='S3')
t.create_node(ParseNode('Bottom0', ''), parent='S1', identifier='B0')
t.create_node(ParseNode('S', 4), parent='S2', identifier='S4')
t.create_node(ParseNode('Front0', ''), parent='S4', identifier='F0')
t.create_node(ParseNode('S', 8), parent='S3', identifier='S5')
t.create_node(ParseNode('Top0', ''), parent='S5', identifier='T0')
  
spatial_model = AoMRSpatialModel()
voxels = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0], 'S3' : [0, 0, 1], 
           'S5' : [-1, 0, 0], 'S4' : [0, 0, 1]}
spatial_model.voxels = voxels
spatial_model._update_positions(t)

ss = AoMRShapeState(forward_model=forward_model, data=data, ll_params=params,
                    spatial_model=spatial_model, initial_tree=t)

# propose new state using add branch move
nss, acc_prob = ss.add_remove_branch_proposal()
nss.tree.show()
print acc_prob
forward_model._view(nss)