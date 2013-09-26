'''
Created on Jun 17, 2013
AoMRShapeGrammar class tests
@author: goker
'''
from treelib import Tree
from pcfg_tree import ParseNode
from aomr_grammar import AoMRShapeState, AoMRSpatialModel
from vision_forward_model import VisionForwardModel
from haptics_forward_model import HapticsForwardModel
import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = np.load('data/haptic/2.npy')
    params = {'b': 9000.0}
    forward_model = HapticsForwardModel(body_fixed=False)
    #h_forward_model = HapticsForwardModel(body_fixed=False)
    
    part1 = 'Bottom0'
    part2 = 'Front0'
    part3 = 'Top0'
    part4 = 'Ear1'
    # wrong parts for this object
    wpart1 = 'Bottom1'
    wpart2 = 'Front1'
    wpart3 = 'Top1'
    wpart4 = 'Ear0'
    
    
    # ======================================================================
    # TEST TREES: We look at the prior, likelihood and acceptance probabilities for
    # different trees. Our purpose is to understand the b value we should set
    # to make sure correct configuration has the highest posterior.
     
    # Tree with 1 part
    t1 = Tree()
    t1.create_node(ParseNode('S', 4), identifier='S')
    t1.create_node(ParseNode('P', 6), identifier='P', parent='S')
    t1.create_node(ParseNode('Body', ''), parent='P')
    
    spatial_model1 = AoMRSpatialModel()
    voxels1 = {'S' : [0,0,0]}
    spatial_model1.voxels = voxels1
    spatial_model1._update_positions(t1)
       
    rrs = AoMRShapeState(forward_model=forward_model, data=data, ll_params=params, spatial_model=spatial_model1, initial_tree=t1)
        
    print rrs
    print ('Prior: %g' % rrs.prior)
    print ('Likelihood: %g' % rrs.likelihood)
    print ('Posterior: %g' % (rrs.prior*rrs.likelihood))
    rrs.tree.show()
     
    # correct tree
    t2 = Tree()
    t2.create_node(ParseNode('S', 7), identifier='S')
    t2.create_node(ParseNode('S', 4), parent='S', identifier='S1')
    t2.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t2.create_node(ParseNode('S', 5), parent='S', identifier='S3')
    t2.create_node(ParseNode('P', 0), identifier='P0', parent='S')
    t2.create_node(ParseNode('Body', ''), identifier='B', parent='P0')
    t2.create_node(ParseNode('P', 3), identifier='P1', parent='S1')
    t2.create_node(ParseNode('P', 1), identifier='P2', parent='S2')
    t2.create_node(ParseNode('P', 5), identifier='P3', parent='S3')
    t2.create_node(ParseNode('S', 4), identifier='S4', parent='S3')
    t2.create_node(ParseNode('P', 7), identifier='P4', parent='S4')
    t2.create_node(ParseNode(part1, ''), parent='P1', identifier='B0')
    t2.create_node(ParseNode(part2, ''), parent='P2', identifier='F0')
    t2.create_node(ParseNode(part3, ''), parent='P3', identifier='T0')
    t2.create_node(ParseNode(part4, ''), parent='P4', identifier='E0')
    
    spatial_model2 = AoMRSpatialModel()
    voxels2 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0], 'S3' : [-1, 0, 1], 
               'S4' : [1, 0, 1]}
    spatial_model2.voxels = voxels2
    spatial_model2._update_positions(t2)
    
    rrs2 = AoMRShapeState(forward_model=forward_model, data=data, ll_params=params, 
                          spatial_model=spatial_model2, initial_tree=t2)
    
    print rrs2
    print ('Prior: %g' % rrs2.prior)
    print ('Likelihood: %g' % rrs2.likelihood)
    print ('Posterior: %g' % (rrs2.prior*rrs2.likelihood))
    rrs2.tree.show()
    
    # bottom, front, top
    t3 = Tree()
    t3.create_node(ParseNode('S', 3), identifier='S')
    t3.create_node(ParseNode('S', 4), parent='S', identifier='S1')
    t3.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t3.create_node(ParseNode('S', 4), parent='S', identifier='S3')
    t3.create_node(ParseNode('P', 0), identifier='P0', parent='S')
    t3.create_node(ParseNode('Body', ''), identifier='B', parent='P0')
    t3.create_node(ParseNode('P', 2), identifier='P1', parent='S1')
    t3.create_node(ParseNode('P', 0), identifier='P2', parent='S2')
    t3.create_node(ParseNode('P', 4), identifier='P3', parent='S3')
    t3.create_node(ParseNode(part1, ''), parent='P1', identifier='B0')
    t3.create_node(ParseNode(part2, ''), parent='P2', identifier='F0')
    t3.create_node(ParseNode(part3, ''), parent='P3', identifier='T0')
       
    spatial_model3 = AoMRSpatialModel()
    voxels3 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0], 'S3' : [-1, 0, 1]}
    spatial_model3.voxels = voxels3
    spatial_model3._update_positions(t3)
    
    rrs3 = AoMRShapeState(forward_model=forward_model, data=data, ll_params=params, 
                          spatial_model=spatial_model3, initial_tree=t3)
    
    print rrs3
    print ('Prior: %g' % rrs3.prior)
    print ('Likelihood: %g' % rrs3.likelihood)
    print ('Posterior: %g' % (rrs3.prior*rrs3.likelihood))
    rrs3.tree.show()
    
    
    # bottom, front
    t4 = Tree()
    t4.create_node(ParseNode('S', 1), identifier='S')
    t4.create_node(ParseNode('S', 4), parent='S', identifier='S1')
    t4.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t4.create_node(ParseNode('P', 0), identifier='P0', parent='S')
    t4.create_node(ParseNode('Body', ''), identifier='B', parent='P0')
    t4.create_node(ParseNode('P', 2), identifier='P1', parent='S1')
    t4.create_node(ParseNode('P', 0), identifier='P2', parent='S2')
    t4.create_node(ParseNode(part1, ''), parent='P1', identifier='B0')
    t4.create_node(ParseNode(part2, ''), parent='P2', identifier='F0')
       
    spatial_model4 = AoMRSpatialModel()
    voxels4 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0]}
    spatial_model4.voxels = voxels4
    spatial_model4._update_positions(t4)
    
    rrs4 = AoMRShapeState(forward_model=forward_model, data=data, ll_params=params, 
                          spatial_model=spatial_model4, initial_tree=t4)
    
    print rrs4
    print ('Prior: %g' % rrs4.prior)
    print ('Likelihood: %g' % rrs4.likelihood)
    print ('Posterior: %g' % (rrs4.prior*rrs4.likelihood))
    rrs4.tree.show()
    
    # an arbitrary tree
    t5 = Tree()
    t5.create_node(ParseNode('S', 3), identifier='S')
    t5.create_node(ParseNode('S', 4), parent='S', identifier='S1')
    t5.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t5.create_node(ParseNode('S', 4), parent='S', identifier='S3')
    t5.create_node(ParseNode('P', 0), identifier='P0', parent='S')
    t5.create_node(ParseNode('Body', ''), identifier='B', parent='P0')
    t5.create_node(ParseNode('P', 2), identifier='P1', parent='S1')
    t5.create_node(ParseNode('P', 0), identifier='P2', parent='S2')
    t5.create_node(ParseNode('P', 4), identifier='P3', parent='S3')
    t5.create_node(ParseNode(part2, ''), parent='P1', identifier='B0')
    t5.create_node(ParseNode(part1, ''), parent='P2', identifier='F0')
    t5.create_node(ParseNode(part3, ''), parent='P3', identifier='T0')
       
    spatial_model5 = AoMRSpatialModel()
    voxels5 = {'S' : [0,0,0], 'S1' : [1, 0, 1], 'S2' : [0, 0, 1], 'S3' : [-1, 0, -1]}
    spatial_model5.voxels = voxels5
    spatial_model5._update_positions(t5)
    
    rrs5 = AoMRShapeState(forward_model=forward_model, data=data, ll_params=params, 
                          spatial_model=spatial_model5, initial_tree=t5)
    rrs5.tree.show()
    
    # correct tree with wrong parts
    t6 = Tree()
    t6.create_node(ParseNode('S', 7), identifier='S')
    t6.create_node(ParseNode('S', 4), parent='S', identifier='S1')
    t6.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t6.create_node(ParseNode('S', 5), parent='S', identifier='S3')
    t6.create_node(ParseNode('P', 0), identifier='P0', parent='S')
    t6.create_node(ParseNode('Body', ''), identifier='B', parent='P0')
    t6.create_node(ParseNode('P', 3), identifier='P1', parent='S1')
    t6.create_node(ParseNode('P', 1), identifier='P2', parent='S2')
    t6.create_node(ParseNode('P', 5), identifier='P3', parent='S3')
    t6.create_node(ParseNode('S', 4), identifier='S4', parent='S3')
    t6.create_node(ParseNode('P', 7), identifier='P4', parent='S4')
    t6.create_node(ParseNode(wpart1, ''), parent='P1', identifier='B0')
    t6.create_node(ParseNode(wpart2, ''), parent='P2', identifier='F0')
    t6.create_node(ParseNode(wpart3, ''), parent='P3', identifier='T0')
    t6.create_node(ParseNode(wpart4, ''), parent='P4', identifier='E0')

    spatial_model6 = AoMRSpatialModel()
    voxels6 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0], 'S3' : [-1, 0, 1], 
               'S4' : [1, 0, 1]}
    spatial_model6.voxels = voxels6
    spatial_model6._update_positions(t6)
    
    rrs6 = AoMRShapeState(forward_model=forward_model, data=data, ll_params=params, 
                          spatial_model=spatial_model6, initial_tree=t6)
    
    print rrs6
    print ('Prior: %g' % rrs6.prior)
    print ('Likelihood: %g' % rrs6.likelihood)
    print ('Posterior: %g' % (rrs6.prior*rrs6.likelihood))
    rrs6.tree.show()
    
    # bottom, front, top
    t7 = Tree()
    t7.create_node(ParseNode('S', 3), identifier='S')
    t7.create_node(ParseNode('S', 4), parent='S', identifier='S1')
    t7.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t7.create_node(ParseNode('S', 4), parent='S', identifier='S3')
    t7.create_node(ParseNode('P', 0), identifier='P0', parent='S')
    t7.create_node(ParseNode('Body', ''), identifier='B', parent='P0')
    t7.create_node(ParseNode('P', 2), identifier='P1', parent='S1')
    t7.create_node(ParseNode('P', 0), identifier='P2', parent='S2')
    t7.create_node(ParseNode('P', 4), identifier='P3', parent='S3')
    t7.create_node(ParseNode(wpart1, ''), parent='P1', identifier='B0')
    t7.create_node(ParseNode(wpart2, ''), parent='P2', identifier='F0')
    t7.create_node(ParseNode(wpart3, ''), parent='P3', identifier='T0')
     
    spatial_model7 = AoMRSpatialModel()
    voxels7 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0], 'S3' : [-1, 0, 1]}
    spatial_model7.voxels = voxels7
    spatial_model7._update_positions(t7)
    
    rrs7 = AoMRShapeState(forward_model=forward_model, data=data, ll_params=params, 
                          spatial_model=spatial_model7, initial_tree=t7)
    
    print rrs7
    print ('Prior: %g' % rrs7.prior)
    print ('Likelihood: %g' % rrs7.likelihood)
    print ('Posterior: %g' % (rrs7.prior*rrs7.likelihood))
    rrs7.tree.show()
    
    
    # bottom, front
    t8 = Tree()
    t8.create_node(ParseNode('S', 1), identifier='S')
    t8.create_node(ParseNode('S', 4), parent='S', identifier='S1')
    t8.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t8.create_node(ParseNode('P', 0), identifier='P0', parent='S')
    t8.create_node(ParseNode('Body', ''), identifier='B', parent='P0')
    t8.create_node(ParseNode('P', 2), identifier='P1', parent='S1')
    t8.create_node(ParseNode('P', 0), identifier='P2', parent='S2')
    t8.create_node(ParseNode(wpart1, ''), parent='P1', identifier='B0')
    t8.create_node(ParseNode(wpart2, ''), parent='P2', identifier='F0')
     
    spatial_model8 = AoMRSpatialModel()
    voxels8 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0]}
    spatial_model8.voxels = voxels8
    spatial_model8._update_positions(t8)
    
    rrs8 = AoMRShapeState(forward_model=forward_model, data=data, ll_params=params, 
                          spatial_model=spatial_model8, initial_tree=t8)
    
    print rrs8
    print ('Prior: %g' % rrs8.prior)
    print ('Likelihood: %g' % rrs8.likelihood)
    print ('Posterior: %g' % (rrs8.prior*rrs8.likelihood))
    rrs8.tree.show()
    
    # ================================================================================
    print('Posteriors')
    print ('1 No parts: %g' % (rrs.prior*rrs.likelihood))
    print ('2 Veridical: %g' % (rrs2.prior*rrs2.likelihood))
    print ('3 Bottom, Front, Top: %g' % (rrs3.prior*rrs3.likelihood))
    print ('4 Bottom, Front: %g' % (rrs4.prior*rrs4.likelihood))
    print ('5 Arbitrary tree: %g' % (rrs5.prior*rrs5.likelihood))
    print ('6 Veridical - wrong parts: %g' % (rrs6.prior*rrs6.likelihood))
    print ('7 Bottom, Front, Top - wrong parts: %g' % (rrs7.prior*rrs7.likelihood))
    print ('8 Bottom, Front - wrong parts: %g' % (rrs8.prior*rrs8.likelihood))
    
    print('Acceptance Probabilities')
    print ('Acceptance Prob 1-2: %f' % rrs._subtree_acceptance_probability(rrs2))
    print ('Acceptance Prob 1-3: %f' % rrs._subtree_acceptance_probability(rrs3))
    print ('Acceptance Prob 1-4: %f' % rrs._subtree_acceptance_probability(rrs4))
    print ('Acceptance Prob 2-3: %f' % rrs2._subtree_acceptance_probability(rrs3))
    print ('Acceptance Prob 2-4: %f' % rrs2._subtree_acceptance_probability(rrs4))
    print ('Acceptance Prob 3-4: %f' % rrs3._subtree_acceptance_probability(rrs4))
#     
#     samples = [rrs, rrs2, rrs3, rrs4, rrs5, rrs6, rrs7, rrs8]
#     n = len(samples)
#     kernel_st_mat = np.zeros((n, n))
#     print('KERNEL VALUES - Subtree kernel')
#     for i in range(n):
#         for j in range(n):
#             kernel_st_mat[i, j] = samples[i].kernel_subtree(samples[j])
#     
#     kernel_st_df = pd.DataFrame(data=kernel_st_mat, index=range(1, n+1), columns=range(1, n+1))
#     print(kernel_st_df)
#     
#     kernel_p_mat = np.zeros((n, n))
#     print('KERNEL VALUES - P-kernel')
#     for i in range(n):
#         for j in range(n):
#             kernel_p_mat[i, j] = samples[i].kernel_p(samples[j])
#     
#     kernel_p_df = pd.DataFrame(data=kernel_p_mat, index=range(1, n+1), columns=range(1, n+1))
#     print(kernel_p_df)
    

