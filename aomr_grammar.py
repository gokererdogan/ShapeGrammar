'''
Analysis of Multisensory Representations 
Shape Grammar Implementation.
This is the grammar we use in the paper.
Created on Jun 4, 2013

@author: goker
'''

from pcfg_tree import PCFG, ParseNode, PCFGTree
from shape_grammar import ShapeGrammarState, SpatialModel
from vision_forward_model import VisionForwardModel
from treelib import Tree
import numpy as np
from copy import deepcopy

"""
Definition of AoMR Probabilistic Context Free Shape Grammar
"""
terminals = ['Front0', 'Front1', 'Bottom0', 'Bottom1', 'Top0', 'Top1', 'Ear0', 'Ear1']
nonterminals = ['S']
start_symbol = 'S'
rules = {'S' : [['S'], ['S', 'S'], ['S', 'S', 'S'], ['S', 'S', 'S', 'S'], 
                ['Front0'], ['Front1'], ['Bottom0'], ['Bottom1'], ['Top0'], ['Top1'], ['Ear0'], ['Ear1']], }
prod_probabilities = {'S' : [1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0]}
# id of rules that produce only terminals for each nonterminal
# this is used for stopping tree from growing without bound and
# enforcing a depth limit
# 
terminating_rule_ids = {'S' : [4, 5, 6, 7, 8, 9, 10, 11]}
# OLD GRAMMAR
# terminals = ['Front0', 'Front1', 'Bottom0', 'Bottom1', 'Top0', 'Top1', 'Ear0', 'Ear1']
# nonterminals = ['S', 'P']
# start_symbol = 'S'
# rules = {'S' : [['P'], ['S'], ['S', 'S'], ['S', 'S', 'S'], ['S', 'S', 'S', 'S']], 
#           'P' : [['Front0'], ['Front1'], ['Bottom0'], ['Bottom1'], ['Top0'], ['Top1'], ['Ear0'], ['Ear1']] }
# prod_probabilities = {'S' : [.2, .2, .2, .2, .2], 'P' : [.125, .125, .125, .125, .125, .125, .125, .125]}
# # id of rules that produce only terminals for each nonterminal
# # this is used for stopping tree from growing without bound and
# # enforcing a depth limit
# # 
# terminating_rule_ids = {'S' : [0], 'P': [0, 1, 2, 3, 4, 5, 6, 7]}

aomr_shape_pcfg = PCFG(terminals, nonterminals, start_symbol, rules, prod_probabilities, terminating_rule_ids)


class AoMRSpatialModel(SpatialModel):
    """
    Spatial Model Class for AoMR Shape Grammar
    Holds voxel 3D ids and positions (origin of voxel)
    for each nonterminal S in tree
    voxels: Holds voxel ids for each S node
        voxels is a dictionary of 3-element voxel ids for terminal
        S nodes (except root) in tree
        A voxel id is of form (x,y,z) where x,y,z \in {-1,0,1}
    positions: Holds positions of origin for each voxel
        positions is a dictionary of 3D vectors for nonterminal S nodes
        Root S node has position 0, 0, 0. Positions for S nodes are
        calculated using voxel ids and width, height, depth of 
        bounding box of shape 
    """
    # width, height, depth of bounding box of shape (used for calculating 
    # part positions)
    width, height, depth = 0.1604, 0.0, 0.0684
    
    def __init__(self, voxels=None, positions=None):
        """
        Initialize spatial model
        If voxels or positions is not given, they are simply instantiated
        with an empty dictionary
        """
        if voxels is None or positions is None:
            self.voxels = {}
            self.positions = {}
        else:
            self.voxels = voxels
            self.positions = positions
        
    
    def _update_positions(self, tree):
        """
        Calculate position for each voxel in tree
        """ 
        # traverse tree depth first and calculate position for
        # each voxel that is not already in positions (already calculated)
        for node in tree.expand_tree(mode=Tree.WIDTH):
            # each S node is a voxel, so we only look for S nodes
            # and we only calculate position for nodes for which we have
            # calculated position before
            if tree[node].tag.symbol in ['S'] and node not in self.positions.keys():
                # if root, voxel position is 0,0,0
                if tree[node].bpointer is None:
                    self.positions[node] = [0.0, 0.0, 0.0]
                else:
                    # find depth of node, we need this for finding voxel width
                    depth = 0
                    nn = node
                    while tree[nn].bpointer is not None:
                        depth = depth + 1
                        nn = tree[nn].bpointer
                    # voxels get smaller by factor of 3 in each dimension as we go deeper
                    # in the tree
                    factor = 3**(depth)
                    voxel_width, voxel_height, voxel_depth = ( self.width / factor, 
                                                               self.height / factor,
                                                               self.depth / factor) 
                    # this voxel's position is calculated from its parent voxel's
                    # position and voxel dimensions
                    x = self.positions[tree[node].bpointer][0] + (self.voxels[node][0] * voxel_width)
                    y = self.positions[tree[node].bpointer][1] + (self.voxels[node][1] * voxel_height)
                    z = self.positions[tree[node].bpointer][2] + (self.voxels[node][2] * voxel_depth)
                    self.positions[node] = [x, y, z]
        
    def update(self, tree, grammar):
        """
        Updates spatial model, removes nodes that are not in
        ''nodes'' parameter from voxels dictionary and samples
        voxel ids for newly added nodes 
        """
        new_nodes = [node for node in tree.expand_tree(mode=Tree.WIDTH) 
                               if tree[node].tag.symbol in ['S']]
        old_nodes = self.voxels.keys()
        removed_nodes = [node for node in old_nodes if node not in new_nodes]
        added_nodes = [node for node in new_nodes if node not in old_nodes]
        
        for n in removed_nodes:
            del self.voxels[n]
            del self.positions[n]
            
        # get random voxel id for added nodes
        for n in added_nodes:
            self.voxels[n] = self._get_random_voxel_id()
        
        self._update_positions(tree)
    
    def propose(self, tree, grammar):
        """
        Proposes a new spatial model based on current one.
        Creates a new spatial model with current voxels,
        updates it, and returns it
        """    
        voxels_copy = deepcopy(self.voxels)
        positions_copy = deepcopy(self.positions)
        proposed_spatial_model = AoMRSpatialModel(voxels_copy, positions_copy)
        proposed_spatial_model.update(tree, grammar)
        return proposed_spatial_model
    
    def _get_random_voxel_id(self):
        """
        Returns a random voxel id 
        """
        # get a random 3d vector with elements \in {-1,0,1}
        return np.random.randint(-1, 2, 3)
        
    def probability(self):
        """
        Returns probability of model
        """
        # each voxel id is uniform sample from 27 (3.3.3) possible values
        # for each item in voxels dictionary we have a voxel id sample as such
        return np.power(1.0 / 27.0, len(self.voxels))

    
class AoMRShapeState(ShapeGrammarState):
    """
    AoMR shape state class for AoMR grammar and spatial model
    """
    def __init__(self, forward_model, data, ll_params, spatial_model, initial_tree=None):
        """
        Constructor for AoMRShapeState
        Note that the first parameter ``grammar`` of base class AoMRShapeState is removed because 
        this class is a grammar specific implementation
        """
        ShapeGrammarState.__init__(self, aomr_shape_pcfg, forward_model, data, ll_params, spatial_model, initial_tree)
    
    # all the other functionality is independent of grammar and spatial model. 
    # hence they are implemented in AoMRShapeState base class
    # grammar and spatial model specific methods are implemented below
    
    def convert_to_parts_positions(self):
        """
        Converts the state representation to parts and positions
        representation that can be given to forward model
        """
        # position for each part is given by its parent S node's
        # position in spatial model, because we have voxels for only
        # S nodes, and parts are simply placed at the center of its
        # parent voxel
        # parent S node of a part is always its parent because
        # of the production rules in the grammar
        parts = []
        positions = []
        # get terminal nodes (parts)
        terminal_nodes = [node for node in self.tree.expand_tree(mode=Tree.WIDTH) 
                    if self.tree[node].tag.symbol in self.grammar.terminals]
        
        for node in terminal_nodes:
            parts.append(self.tree[node].tag.symbol)
            # parent S node of a part is always its parent's parent because
            # of the production rules in the grammar
            snode = self.tree[node].bpointer
            positions.append(self.spatial_model.positions[snode])
        
        return parts, positions
    
#     def _prior(self):
#         """
#         Override PCFGTree prior, we do not integrate
#         out production probabilities as it is unnecessary
#         in our case (since all our uniform) and it decreases
#         prior probabilities for nothing
#         Then in our case prior probability becomes only
#         the derivation probability for a parse tree and
#         spatial model's prior probability
#         """
#         return PCFGTree._derivation_prob(self) * self.spatial_model.probability()
#     
    def acceptance_prob(self, proposal):
        """
        Acceptance probability for AoMR Shape Grammar is simply 
        Rational Rules acceptance probability. The derivation for
        this can be seen in the paper.
        """
        nt_current = [node for node in self.tree.expand_tree(mode=Tree.WIDTH) 
                           if self.tree[node].tag.symbol in self.grammar.nonterminals]
        nt_proposal = [node for node in proposal.tree.expand_tree(mode=Tree.WIDTH) 
                           if proposal.tree[node].tag.symbol in self.grammar.nonterminals]
        acc_prob = 1
        acc_prob = acc_prob * proposal.prior * proposal.likelihood * len(nt_current) * self.derivation_prob * self.spatial_model.probability()
        acc_prob = acc_prob / (self.prior * self.likelihood * len(nt_proposal) * proposal.derivation_prob * proposal.spatial_model.probability()) 
        return acc_prob
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        
        self_parts_pos = []
        other_parts_pos = []
        parts, positions = self.convert_to_parts_positions()
        for part, pos in zip(parts, positions):
            self_parts_pos.append([part, pos])
            
        parts, positions = other.convert_to_parts_positions()
        for part, pos in zip(parts, positions):
            other_parts_pos.append([part, pos])
            
        self_parts_pos.sort()
        other_parts_pos.sort()
        return self_parts_pos == other_parts_pos
    
    def __neq__(self, other):
        return not self.__eq__(other)
    
    def __repr__(self):
        parts, positions = self.convert_to_parts_positions()
        return "".join('(' + part + ')(' + repr(pos) + ')' for part, pos in zip(parts, positions) )
        
    def __str__(self):
        return self.__repr__()
    
    
        

if __name__ == '__main__':
    data = np.load('data/visual/1.npy')
    params = {'b': 1250.0}
    forward_model = VisionForwardModel()
    #forward_model = HapticsForwardModel()
    
#     # RANDOM STATE
#     spatial_model = AoMRSpatialModel()
#     rs = AoMRShapeState(forward_model, data, params, spatial_model)
#     print rs
#     print ('Prior: %g' % rs.prior)
#     print ('Likelihood: %g' % rs.likelihood)
#     print ('Posterior: %g' % (rs.prior*rs.likelihood))
#     rs.tree.show()
#      
#     forward_model._view(rs)
#     
    # TEST TREES: We look at the prior, likelihood and acceptance probabilities for
    # empty tree, correct configuration (4 parts in correct positions) and tree with 
    # 1 part (ear) removed. Our purpose is to understand the b value we should set
    # to make sure correct configuration has the highest posterior.
    
    
    # Tree with 1 part
    t1 = Tree()
    t1.create_node(ParseNode('S', 8), identifier='S')
    t1.create_node(ParseNode('Top0', ''), parent='S')
     
    spatial_model1 = AoMRSpatialModel()
    voxels1 = {'S' : [0,0,0]}
    spatial_model1.voxels = voxels1
    spatial_model1._update_positions(t1)
    
    rrs = AoMRShapeState(forward_model, data, params, spatial_model1, t1)
     
    print rrs
    print ('Prior: %g' % rrs.prior)
    print ('Likelihood: %g' % rrs.likelihood)
    print ('Posterior: %g' % (rrs.prior*rrs.likelihood))
    rrs.tree.show()
     
    rules = {'S' : [['S'], ['S', 'S'], ['S', 'S', 'S'], ['S', 'S', 'S', 'S'], 
                ['Front0'], ['Front1'], ['Bottom0'], ['Bottom1'], ['Top0'], ['Top1'], ['Ear0'], ['Ear1']], }

    # correct tree
    t2 = Tree()
    t2.create_node(ParseNode('S', 2), identifier='S')
    t2.create_node(ParseNode('S', 6), parent='S', identifier='S1')
    t2.create_node(ParseNode('S', 0), parent='S', identifier='S2')
    t2.create_node(ParseNode('S', 1), parent='S', identifier='S3')
    t2.create_node(ParseNode('Bottom0', ''), parent='S1', identifier='B0')
    t2.create_node(ParseNode('S', 4), parent='S2', identifier='S4')
    t2.create_node(ParseNode('Front0', ''), parent='S4', identifier='F0')
    t2.create_node(ParseNode('S', 8), parent='S3', identifier='S5')
    t2.create_node(ParseNode('S', 10), parent='S3', identifier='S6')
    t2.create_node(ParseNode('Top0', ''), parent='S5', identifier='T0')
    t2.create_node(ParseNode('Ear0', ''), parent='S6', identifier='E0')
      
    spatial_model2 = AoMRSpatialModel()
    voxels2 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0], 'S3' : [0, 0, 1], 
               'S5' : [-1, 0, 0], 'S6' : [-1, 0, 1], 'S4' : [0, 0, 1]}
    spatial_model2.voxels = voxels2
    spatial_model2._update_positions(t2)
      
    rrs2 = AoMRShapeState(forward_model, data, params, spatial_model2, t2)
    print rrs2
    print ('Prior: %g' % rrs2.prior)
    print ('Likelihood: %g' % rrs2.likelihood)
    print ('Posterior: %g' % (rrs2.prior*rrs2.likelihood))
    rrs2.tree.show()
      
    print ('Acceptance Prob 1-2: %f' % rrs.acceptance_prob(rrs2))
       
    # tree with 1 part missing
    t3 = Tree()
    t3.create_node(ParseNode('S', 2), identifier='S')
    t3.create_node(ParseNode('S', 6), parent='S', identifier='S1')
    t3.create_node(ParseNode('S', 0), parent='S', identifier='S2')
    t3.create_node(ParseNode('S', 0), parent='S', identifier='S3')
    t3.create_node(ParseNode('Bottom0', ''), parent='S1', identifier='B0')
    t3.create_node(ParseNode('S', 4), parent='S2', identifier='S4')
    t3.create_node(ParseNode('Front0', ''), parent='S4', identifier='F0')
    t3.create_node(ParseNode('S', 8), parent='S3', identifier='S5')
    t3.create_node(ParseNode('Top0', ''), parent='S5', identifier='T0')
      
    spatial_model3 = AoMRSpatialModel()
    voxels3 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0], 'S3' : [0, 0, 1], 
               'S5' : [-1, 0, 0], 'S4' : [0, 0, 1]}
    spatial_model3.voxels = voxels3
    spatial_model3._update_positions(t3)
           
    rrs3 = AoMRShapeState(forward_model, data, params, spatial_model3, t3)
    print rrs3
    print ('Prior: %g' % rrs3.prior)
    print ('Likelihood: %g' % rrs3.likelihood)
    print ('Posterior: %g' % (rrs3.prior*rrs3.likelihood))
    rrs3.tree.show()
     
       
    print ('Acceptance Prob 1-3: %f' % rrs.acceptance_prob(rrs3))
    print ('Acceptance Prob 2-3: %f' % rrs2.acceptance_prob(rrs3))
#       
#     # tree with only bottom and ear
#     t4 = Tree()
#     t4.create_node(ParseNode('S', 3), identifier='S')
#     t4.create_node(ParseNode('S', 0), parent='S', identifier='S1')
#     t4.create_node(ParseNode('S', 1), parent='S', identifier='S3')
#     t4.create_node(ParseNode('P', 2), parent='S1', identifier='P1')
#     t4.create_node(ParseNode('Bottom0', ''), parent='P1', identifier='B0')
#     t4.create_node(ParseNode('S', 0), parent='S3', identifier='S5')
#     t4.create_node(ParseNode('P', 6), parent='S5', identifier='P4')
#     t4.create_node(ParseNode('Ear0', ''), parent='P4', identifier='E0')
#     
#     spatial_model4 = AoMRSpatialModel()
#     voxels4 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S3' : [0, 0, 1], 
#                'S5' : [-1, 0, 1]}
#     spatial_model4.voxels = voxels4
#     spatial_model4._update_positions(t4)
#       
#     rrs4 = AoMRShapeState(forward_model, data, params, spatial_model4, t4)
#     print rrs4
#     print ('Prior: %g' % rrs4.prior)
#     print ('Likelihood: %g' % rrs4.likelihood)
#     print ('Posterior: %g' % (rrs4.prior*rrs4.likelihood))
#     rrs4.tree.show()
#       
#     print ('Acceptance Prob 1-4: %f' % rrs.acceptance_prob(rrs4))
#     print ('Acceptance Prob 2-4: %f' % rrs2.acceptance_prob(rrs4))
#     print ('Acceptance Prob 3-4: %f' % rrs3.acceptance_prob(rrs4))
#     
    # tree with only bottom
    t5 = Tree()
    t5.create_node(ParseNode('S', 0), identifier='S')
    t5.create_node(ParseNode('S', 0), parent='S', identifier='S1')
    t5.create_node(ParseNode('Bottom0', ''), parent='S1', identifier='B0')
     
    spatial_model5 = AoMRSpatialModel()
    voxels5 = {'S' : [0,0,0], 'S1' : [0, 0, -1]}
    spatial_model5.voxels = voxels5
    spatial_model5._update_positions(t5)
       
    rrs5 = AoMRShapeState(forward_model, data, params, spatial_model5, t5)
    print rrs5
    print ('Prior: %g' % rrs5.prior)
    print ('Likelihood: %g' % rrs5.likelihood)
    print ('Posterior: %g' % (rrs5.prior*rrs5.likelihood))
    rrs5.tree.show()
       
    print ('Acceptance Prob 1-5: %f' % rrs.acceptance_prob(rrs5))
    print ('Acceptance Prob 2-5: %f' % rrs2.acceptance_prob(rrs5))
    print ('Acceptance Prob 3-5: %f' % rrs3.acceptance_prob(rrs5))
#     print ('Acceptance Prob 4-5: %f' % rrs4.acceptance_prob(rrs5))
#     
    # tree with only bottom and front
    t6 = Tree()
    t6.create_node(ParseNode('S', 1), identifier='S')
    t6.create_node(ParseNode('S', 6), parent='S', identifier='S1')
    t6.create_node(ParseNode('S', 0), parent='S', identifier='S2')
    t6.create_node(ParseNode('Bottom0', ''), parent='S1', identifier='B0')
    t6.create_node(ParseNode('S', 4), parent='S2', identifier='S4')
    t6.create_node(ParseNode('Front0', ''), parent='S4', identifier='F0')
     
    spatial_model6 = AoMRSpatialModel()
    voxels6 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0], 'S4' : [0, 0, 1]}
    spatial_model6.voxels = voxels6
    spatial_model6._update_positions(t6)
           
    rrs6 = AoMRShapeState(forward_model, data, params, spatial_model6, t6)
    print rrs6
    print ('Prior: %g' % rrs6.prior)
    print ('Likelihood: %g' % rrs6.likelihood)
    print ('Posterior: %g' % (rrs6.prior*rrs6.likelihood))
    rrs6.tree.show()
     
       
    print ('Acceptance Prob 1-6: %f' % rrs.acceptance_prob(rrs6))
    print ('Acceptance Prob 2-6: %f' % rrs2.acceptance_prob(rrs6))
    print ('Acceptance Prob 3-6: %f' % rrs3.acceptance_prob(rrs6))
#     print ('Acceptance Prob 4-6: %f' % rrs4.acceptance_prob(rrs6))
    print ('Acceptance Prob 5-6: %f' % rrs5.acceptance_prob(rrs6))
     
    # front part slightly mislocated
    t7 = Tree()
    t7.create_node(ParseNode('S', 2), identifier='S')
    t7.create_node(ParseNode('S', 6), parent='S', identifier='S1')
    t7.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t7.create_node(ParseNode('S', 1), parent='S', identifier='S3')
    t7.create_node(ParseNode('Bottom0', ''), parent='S1', identifier='B0')
    t7.create_node(ParseNode('Front0', ''), parent='S2', identifier='F0')
    t7.create_node(ParseNode('S', 8), parent='S3', identifier='S5')
    t7.create_node(ParseNode('S', 10), parent='S3', identifier='S6')
    t7.create_node(ParseNode('Top0', ''), parent='S5', identifier='T0')
    t7.create_node(ParseNode('Ear0', ''), parent='S6', identifier='E0')
      
    spatial_model7 = AoMRSpatialModel()
    voxels7 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0], 'S3' : [0, 0, 1], 
               'S5' : [-1, 0, 0], 'S6' : [-1, 0, 1]}
    spatial_model7.voxels = voxels7
    spatial_model7._update_positions(t7)
       
    rrs7 = AoMRShapeState(forward_model, data, params, spatial_model7, t7)
    print rrs7
    print ('Prior: %g' % rrs7.prior)
    print ('Likelihood: %g' % rrs7.likelihood)
    print ('Posterior: %g' % (rrs7.prior*rrs7.likelihood))
    rrs7.tree.show()
     
    print ('Acceptance Prob 1-7: %f' % rrs.acceptance_prob(rrs7))
    print ('Acceptance Prob 2-7: %f' % rrs2.acceptance_prob(rrs7))
    print ('Acceptance Prob 3-7: %f' % rrs3.acceptance_prob(rrs7))
#     print ('Acceptance Prob 4-7: %f' % rrs4.acceptance_prob(rrs7))
    print ('Acceptance Prob 5-7: %f' % rrs5.acceptance_prob(rrs7))
    print ('Acceptance Prob 6-7: %f' % rrs6.acceptance_prob(rrs7))
     
    forward_model._view(rrs7)
    
    print('Posteriors')
    print ('1 No parts: %g' % (rrs.prior*rrs.likelihood))
    print ('2 Veridical: %g' % (rrs2.prior*rrs2.likelihood))
    print ('3 Bottom, Front, Top: %g' % (rrs3.prior*rrs3.likelihood))
    print ('4 Bottom: %g' % (rrs5.prior*rrs5.likelihood))
    print ('5 Bottom, Front: %g' % (rrs6.prior*rrs6.likelihood))
    print ('6 Mislocated: %g' % (rrs7.prior*rrs7.likelihood))
    
    print('Acceptance Probabilities')
    print ('Acceptance Prob 1-2: %f' % rrs.acceptance_prob(rrs2))
    print ('Acceptance Prob 1-3: %f' % rrs.acceptance_prob(rrs3))
    print ('Acceptance Prob 1-4: %f' % rrs.acceptance_prob(rrs5))
    print ('Acceptance Prob 1-5: %f' % rrs.acceptance_prob(rrs6))
    print ('Acceptance Prob 1-6: %f' % rrs.acceptance_prob(rrs7))
    print ('Acceptance Prob 2-3: %f' % rrs2.acceptance_prob(rrs3))
    print ('Acceptance Prob 2-4: %f' % rrs2.acceptance_prob(rrs5))
    print ('Acceptance Prob 2-5: %f' % rrs2.acceptance_prob(rrs6))
    print ('Acceptance Prob 2-6: %f' % rrs2.acceptance_prob(rrs7))
    print ('Acceptance Prob 3-4: %f' % rrs3.acceptance_prob(rrs5))
    print ('Acceptance Prob 3-5: %f' % rrs3.acceptance_prob(rrs6))
    print ('Acceptance Prob 3-6: %f' % rrs3.acceptance_prob(rrs7))
    print ('Acceptance Prob 4-5: %f' % rrs5.acceptance_prob(rrs6))
    print ('Acceptance Prob 4-6: %f' % rrs5.acceptance_prob(rrs7))
    print ('Acceptance Prob 5-6: %f' % rrs6.acceptance_prob(rrs7))