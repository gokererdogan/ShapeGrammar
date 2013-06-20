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
    width, height, depth = 0.12, 0.0, 0.0684
    
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
    def __init__(self, forward_model=None, data=None, ll_params=None, spatial_model=None, initial_tree=None):
        """
        Constructor for AoMRShapeState
        Note that the first parameter ``grammar`` of base class AoMRShapeState is removed because 
        this class is a grammar specific implementation
        """
        self.moves = [self.subtree_proposal, self.add_remove_branch_proposal, 
                      self.change_part_proposal, self.refine_part_position_proposal]
        ShapeGrammarState.__init__(self, grammar=aomr_shape_pcfg, forward_model=forward_model, 
                                   data=data, ll_params=ll_params, spatial_model=spatial_model, 
                                   initial_tree=initial_tree)
    
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
    
    
    def add_remove_branch_proposal(self):
        """
        Proposes a new state based on current state using add/remove branch move
        First choose randomly if we are doing an add or remove move
        If add move: finds all S nodes with less than 4 children, chooses
            one randomly, chooses a random part to append to it and adds a child
            S->P node to chosen node
            This move essentially adds a new part to object
        If remove move: finds all S nodes that directly go to parts, chooses one
            randomly and removes it.
        """
        proposal_tree = deepcopy(self.tree)
        which_move = np.random.rand()
        move_type = -1
        if which_move < .5: # add move
            move_type = 0
            # find a suitable S node
            suitable_nodes = self._add_remove_branch_get_suitable_nodes(proposal_tree, move_type)
            
            # if there are nodes to choose from, else do nothing
            if len(suitable_nodes) > 0:
                # choose one of the suitable nodes randomly
                chosen_node_id = np.random.choice(suitable_nodes)
                
                # sample a new part for the new S node
                chosen_part_rule_id = np.random.choice(self.grammar.terminating_rule_ids['S'])
                chosen_part = self.grammar.rules['S'][chosen_part_rule_id][0]
                
                # add the new S branch and its part
                new_snode = proposal_tree.create_node(tag=ParseNode('S', chosen_part_rule_id), parent=chosen_node_id)
                proposal_tree.create_node(tag=ParseNode(chosen_part, ''), parent=new_snode.identifier)
                
                # change the production rule used in node's parent
                proposal_tree[chosen_node_id].tag.rule = proposal_tree[chosen_node_id].tag.rule + 1
        else: # remove move
            move_type = 1
            suitable_remove_nodes = self._add_remove_branch_get_suitable_nodes(proposal_tree, move_type)
            
            # if we have nodes to choose from, else do nothing
            if len(suitable_remove_nodes) > 0:
                # choose one of the suitable nodes for removal
                chosen_node_id = np.random.choice(suitable_remove_nodes)
                parent_node_id = proposal_tree[chosen_node_id].bpointer
                # remove node
                proposal_tree.remove_node(chosen_node_id)
                
                # change the production rule used in its parent
                proposal_tree[parent_node_id].tag.rule = proposal_tree[parent_node_id].tag.rule - 1
            
        # get a new spatial model based on proposed tree
        proposed_spatial_model = self.spatial_model.propose(proposal_tree, self.grammar)
        proposal = self.__class__(forward_model=self.forward_model, data=self.data, ll_params=self.ll_params, 
                                  spatial_model=proposed_spatial_model, initial_tree=proposal_tree)
        
        # get acceptance probability
        acc_prob = self._add_remove_branch_acceptance_probability(proposal, move_type)
        return proposal, acc_prob    
    
    def _add_remove_branch_get_suitable_nodes(self, proposal_tree, move_type):
        """
        Gets nodes that can be removed or added for add/branch move depending
        on move_type parameter
        """
        suitable_nodes = []
        if move_type == 0: # add move
            for node in proposal_tree.expand_tree(mode=Tree.WIDTH):
                # if node is an S node and has
                # less than 3 children
                # NOTE THIS CODE IS GRAMMAR SPECIFIC, BE CAREFUL WHEN YOU
                # USE THIS PROPOSAL WITH A DIFFERENT GRAMMAR
                if proposal_tree[node].tag.symbol == 'S' and proposal_tree[node].tag.rule < 3:
                    suitable_nodes.append(node)
        elif move_type == 1: # remove move
            # find nodes that can be removed
            for node in proposal_tree.expand_tree(mode=Tree.WIDTH):
                # if node is not root, is an S node and directly goes to a part
                # and is not the only child of its parent
                if proposal_tree[node].bpointer is not None and \
                    proposal_tree[node].tag.symbol == 'S' and \
                    proposal_tree[node].tag.rule > 3 and \
                    len(proposal_tree[proposal_tree[node].bpointer].fpointer) > 1:
                    suitable_nodes.append(node)
        else:
            raise ValueError('move_type can only be 0 or 1')
        
        return suitable_nodes
    
    def _add_remove_branch_acceptance_probability(self, proposal, move_type):
        """
        Acceptance probability for add/remove branch move
        """
        # proposal probabilities
        q_sp_s = 1
        q_s_sp = 1
        if move_type == 0: # add move
            # number of branches we can remove in proposal
            removable_nodes = self._add_remove_branch_get_suitable_nodes(proposal.tree, 1)
            # nodes to which we can add a branch in current state
            add_nodes = self._add_remove_branch_get_suitable_nodes(self.tree, 0)
            if len(removable_nodes) > 0 and len(add_nodes) > 0:
                # q(S' -> S) or q(S|S')
                q_sp_s = (1.0 / len(removable_nodes))
                # q(S -> S') or q(S'|S)
                q_s_sp = (1.0 / len(add_nodes)) * (1.0 / 8.0)
        elif move_type == 1: # remove move
            # nodes to which we can add a branch in proposal
            add_nodes = self._add_remove_branch_get_suitable_nodes(proposal.tree, 0)
            # nodes we can remove in current state
            removable_nodes = self._add_remove_branch_get_suitable_nodes(self.tree, 1)
            if len(removable_nodes) > 0 and len(add_nodes) > 0:
                # q(S' -> S) or q(S|S')
                q_sp_s = (1.0 / len(add_nodes)) * (1.0 / 8.0)
                # q(S -> S') or q(S'|S)
                q_s_sp = (1.0 / len(removable_nodes))
        else:
            raise ValueError('move_type can only be 0 or 1')
        
        acc_prob = 1
        
        # prior terms contain prior probabilities for spatial model too, so
        # in order to get back to Rational Rules prior we multiply with
        # inverse of spatial model probabilities
        acc_prob = acc_prob * proposal.prior * proposal.likelihood * q_sp_s * self.spatial_model.probability()
        acc_prob = acc_prob / (self.prior * self.likelihood * q_s_sp * proposal.spatial_model.probability())
        return acc_prob
        
    def change_part_proposal(self):
        """
        Change part move. Select one of the parts in the object
        randomly and changes it to another part.
        """
        proposal_tree = deepcopy(self.tree)
        proposal_spatial_model = deepcopy(self.spatial_model)
        
        # get all the part nodes
        part_nodes = []
        for node in proposal_tree.expand_tree(mode=Tree.WIDTH): 
            if proposal_tree[node].tag.symbol in self.grammar.terminals:
                part_nodes.append(node)
                
        # choose one node part randomly
        chosen_node = np.random.choice(part_nodes)
        # choose a random part to replace it
        new_part = np.random.choice(self.grammar.terminals)
        # change part to new part and update its parent's rule number
        proposal_tree[chosen_node].tag.symbol = new_part
        proposal_tree[proposal_tree[chosen_node].bpointer].tag.rule = self.grammar.rules['S'].index([new_part])
        proposal = self.__class__(forward_model=self.forward_model, data=self.data, ll_params=self.ll_params, 
                                  spatial_model=proposal_spatial_model, initial_tree=proposal_tree)
        
        # prior terms contain prior probabilities for spatial model too, so
        # in order to get back to Rational Rules prior we multiply with
        # inverse of spatial model probabilities
        # NOTE: this move does not change the prior since proposal is symmetric.
        acc_prob = 1
        acc_prob = acc_prob * proposal.prior * proposal.likelihood * self.spatial_model.probability()
        acc_prob = acc_prob / (self.prior * self.likelihood * proposal.spatial_model.probability())
        return proposal, acc_prob    
    
    def refine_part_position_proposal(self):
        """
        Proposes a new state based on current state using refine part position move
        First choose randomly if we are doing an add or remove move
        If add move: finds all part nodes, choose one randomly and add a new S
            node as its parent. 
            This move refines the position of part by moving it one level deeper in 
            tree, hence specifying its location in a finer level of granularity
        If remove move: finds all P nodes whose father S node is the only child of
            its parent. Choose one randomly and remove its parent node. 
            This moves corresponds to specifying the position of the part in a 
            coarser level of granularity
        """
        proposal_tree = deepcopy(self.tree)
        which_move = np.random.rand()
        move_type = -1
        if which_move < .5: # add move
            move_type = 0
            # find a suitable S node
            suitable_nodes = self._refine_part_position_get_suitable_nodes(proposal_tree, move_type)
            
            # if there are nodes to choose from, else do nothing
            if len(suitable_nodes) > 0:
                # choose one of the suitable nodes randomly
                chosen_node_id = np.random.choice(suitable_nodes)
                parent_id = proposal_tree[chosen_node_id].bpointer
                rule = proposal_tree[parent_id].tag.rule
                
                # add the new S branch and connect the part to it
                new_snode = proposal_tree.create_node(tag=ParseNode('S', rule), parent=parent_id)
                proposal_tree.move_node(chosen_node_id, new_snode.identifier)
                
                # change the production rule used in node's grandparent
                proposal_tree[parent_id].tag.rule = 0
        else: # remove move
            move_type = 1
            suitable_remove_nodes = self._refine_part_position_get_suitable_nodes(proposal_tree, move_type)
            
            # if we have nodes to choose from, else do nothing
            if len(suitable_remove_nodes) > 0:
                # choose one of the suitable nodes for removal
                chosen_node_id = np.random.choice(suitable_remove_nodes)
                parent_node_id = proposal_tree[chosen_node_id].bpointer
                grandparent_node_id = proposal_tree[parent_node_id].bpointer
                
                # connect part node to its grandparent
                proposal_tree.move_node(chosen_node_id, grandparent_node_id)
                # change the production rule used in its grandparent
                proposal_tree[grandparent_node_id].tag.rule = proposal_tree[parent_node_id].tag.rule
                # remove S node
                proposal_tree.remove_node(parent_node_id)
            
        # get a new spatial model based on proposed tree
        proposed_spatial_model = self.spatial_model.propose(proposal_tree, self.grammar)
        proposal = self.__class__(forward_model=self.forward_model, data=self.data, ll_params=self.ll_params, 
                                  spatial_model=proposed_spatial_model, initial_tree=proposal_tree)
        
        # get acceptance probability
        acc_prob = self._refine_part_position_acceptance_probability(proposal, move_type)
        return proposal, acc_prob
    
    def _refine_part_position_get_suitable_nodes(self, proposal_tree, move_type):
        """
        Gets nodes that can be removed or added for refine part position move depending
        on move_type parameter
        """
        suitable_nodes = []
        if move_type == 0: # add move
            for node in proposal_tree.expand_tree(mode=Tree.WIDTH):
                # if node is a terminal
                if proposal_tree[node].tag.symbol in self.grammar.terminals:
                    suitable_nodes.append(node)
        elif move_type == 1: # remove move
            # find nodes that can be removed
            for node in proposal_tree.expand_tree(mode=Tree.WIDTH):
                # if node is a leaf and it grandparent has only one child
                if proposal_tree[node].tag.symbol in self.grammar.terminals and \
                    proposal_tree[proposal_tree[node].bpointer].bpointer is not None and \
                    len(proposal_tree[proposal_tree[proposal_tree[node].bpointer].bpointer].fpointer) == 1:
                    suitable_nodes.append(node)
        else:
            raise ValueError('move_type can only be 0 or 1')
        
        return suitable_nodes
    
    def _refine_part_position_acceptance_probability(self, proposal, move_type):
        """
        Acceptance probability for refine part position move
        """
        # proposal probabilities
        q_sp_s = 1
        q_s_sp = 1
        if move_type == 0: # add move
            # number of branches we can remove in proposal
            removable_nodes = self._refine_part_position_get_suitable_nodes(proposal.tree, 1)
            # nodes to which we can add a branch in current state
            add_nodes = self._refine_part_position_get_suitable_nodes(self.tree, 0)
            if len(removable_nodes) > 0 and len(add_nodes) > 0:
                # q(S' -> S) or q(S|S')
                q_sp_s = (1.0 / len(removable_nodes))
                # q(S -> S') or q(S'|S)
                q_s_sp = (1.0 / len(add_nodes))
        elif move_type == 1: # remove move
            # nodes to which we can add a branch in proposal
            add_nodes = self._refine_part_position_get_suitable_nodes(proposal.tree, 0)
            # nodes we can remove in current state
            removable_nodes = self._refine_part_position_get_suitable_nodes(self.tree, 1)
            if len(removable_nodes) > 0 and len(add_nodes) > 0:
                # q(S' -> S) or q(S|S')
                q_sp_s = (1.0 / len(add_nodes))
                # q(S -> S') or q(S'|S)
                q_s_sp = (1.0 / len(removable_nodes))
        else:
            raise ValueError('move_type can only be 0 or 1')
        
        acc_prob = 1
        
        # prior terms contain prior probabilities for spatial model too, so
        # in order to get back to Rational Rules prior we multiply with
        # inverse of spatial model probabilities
        acc_prob = acc_prob * proposal.prior * proposal.likelihood * q_sp_s * self.spatial_model.probability()
        acc_prob = acc_prob / (self.prior * self.likelihood * q_s_sp * proposal.spatial_model.probability())
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
        return "".join('({0:s})(x: {1:.4f}, y: {2:.4f}, z: {3:.4f})'.format(part, pos[0], pos[1], pos[2]) for part, pos in zip(parts, positions) )
        
    def __str__(self):
        return self.__repr__()
    
    
        

if __name__ == '__main__':
    data = np.load('data/visual/1.npy')
    params = {'b': 1600.0}
    forward_model = VisionForwardModel()
    #forward_model = HapticsForwardModel()
    
    part1 = 'Bottom0'
    part2 = 'Front0'
    part3 = 'Top0'
    part4 = 'Ear0'
    
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
    # different trees. Our purpose is to understand the b value we should set
    # to make sure correct configuration has the highest posterior.
    
     
    # Tree with 1 part
    t1 = Tree()
    t1.create_node(ParseNode('S', 8), identifier='S')
    t1.create_node(ParseNode(part4, ''), parent='S')
    
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
    t2.create_node(ParseNode('S', 2), identifier='S')
    t2.create_node(ParseNode('S', 6), parent='S', identifier='S1')
    t2.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t2.create_node(ParseNode('S', 1), parent='S', identifier='S3')
    t2.create_node(ParseNode(part1, ''), parent='S1', identifier='B0')
    t2.create_node(ParseNode(part2, ''), parent='S2', identifier='F0')
    t2.create_node(ParseNode('S', 8), parent='S3', identifier='S5')
    t2.create_node(ParseNode('S', 10), parent='S3', identifier='S6')
    t2.create_node(ParseNode(part3, ''), parent='S5', identifier='T0')
    t2.create_node(ParseNode(part4, ''), parent='S6', identifier='E0')
       
    spatial_model2 = AoMRSpatialModel()
    voxels2 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0], 'S3' : [-1, 0, 1], 
               'S5' : [0, 0, 0], 'S6' : [1, 0, 1]}
    spatial_model2.voxels = voxels2
    spatial_model2._update_positions(t2)
    
    rrs2 = AoMRShapeState(forward_model=forward_model, data=data, ll_params=params, spatial_model=spatial_model2, initial_tree=t2)
    
    print rrs2
    print ('Prior: %g' % rrs2.prior)
    print ('Likelihood: %g' % rrs2.likelihood)
    print ('Posterior: %g' % (rrs2.prior*rrs2.likelihood))
    rrs2.tree.show()
    
#     if rrs2.likelihood < 1:
#         render = rrs2.render
#         import matplotlib.pyplot as pl
#         pl.figure()
#         pl.subplot(2,3,1)
#         pl.imshow(data[0,:,:], cmap='gray')
#         pl.subplot(2,3,2)
#         pl.imshow(data[1,:,:], cmap='gray')
#         pl.subplot(2,3,3)
#         pl.imshow(data[2,:,:], cmap='gray')
#         pl.subplot(2,3,4)
#         pl.imshow(render[0,:,:], cmap='gray')
#         pl.subplot(2,3,5)
#         pl.imshow(render[1,:,:], cmap='gray')
#         pl.subplot(2,3,6)
#         pl.imshow(render[2,:,:], cmap='gray')
#         pl.show()


    # bottom, front, top
    t3 = Tree()
    t3.create_node(ParseNode('S', 2), identifier='S')
    t3.create_node(ParseNode('S', 6), parent='S', identifier='S1')
    t3.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t3.create_node(ParseNode('S', 8), parent='S', identifier='S3')
    t3.create_node(ParseNode(part1, ''), parent='S1', identifier='B0')
    t3.create_node(ParseNode(part2, ''), parent='S2', identifier='F0')
    t3.create_node(ParseNode(part3, ''), parent='S3', identifier='T0')
       
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
    
    # bottom, front, top
    t4 = Tree()
    t4.create_node(ParseNode('S', 1), identifier='S')
    t4.create_node(ParseNode('S', 6), parent='S', identifier='S1')
    t4.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t4.create_node(ParseNode(part1, ''), parent='S1', identifier='B0')
    t4.create_node(ParseNode(part2, ''), parent='S2', identifier='F0')
       
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
    
    # bottom, front, top with tree structure ready for ear part
    t5 = Tree()
    t5.create_node(ParseNode('S', 2), identifier='S')
    t5.create_node(ParseNode('S', 6), parent='S', identifier='S1')
    t5.create_node(ParseNode('S', 4), parent='S', identifier='S2')
    t5.create_node(ParseNode('S', 0), parent='S', identifier='S3')
    t5.create_node(ParseNode(part1, ''), parent='S1', identifier='B0')
    t5.create_node(ParseNode(part2, ''), parent='S2', identifier='F0')
    t5.create_node(ParseNode('S', 8), parent='S3', identifier='S5')
    t5.create_node(ParseNode(part3, ''), parent='S5', identifier='T0')
       
    spatial_model5 = AoMRSpatialModel()
    voxels5 = {'S' : [0,0,0], 'S1' : [0, 0, -1], 'S2' : [1, 0, 0], 'S3' : [-1, 0, 1], 
               'S5' : [0, 0, 0]}
    spatial_model5.voxels = voxels5
    spatial_model5._update_positions(t5)
    
    rrs5 = AoMRShapeState(forward_model=forward_model, data=data, ll_params=params, 
                          spatial_model=spatial_model5, initial_tree=t5)
    
    print rrs5
    print ('Prior: %g' % rrs5.prior)
    print ('Likelihood: %g' % rrs5.likelihood)
    print ('Posterior: %g' % (rrs5.prior*rrs5.likelihood))
    rrs5.tree.show()
     
    print('Posteriors')
    print ('1 No parts: %g' % (rrs.prior*rrs.likelihood))
    print ('2 Veridical: %g' % (rrs2.prior*rrs2.likelihood))
    print ('3 Bottom, Front, Top: %g' % (rrs3.prior*rrs3.likelihood))
    print ('4 Bottom, Front: %g' % (rrs4.prior*rrs4.likelihood))
    print ('5 Bottom, Front, Top, (Ear): %g' % (rrs5.prior*rrs5.likelihood))
    
    print('Acceptance Probabilities')
    print ('Acceptance Prob 1-2: %f' % rrs._subtree_acceptance_probability(rrs2))
    print ('Acceptance Prob 1-3: %f' % rrs._subtree_acceptance_probability(rrs3))
    print ('Acceptance Prob 1-4: %f' % rrs._subtree_acceptance_probability(rrs4))
    print ('Acceptance Prob 1-5: %f' % rrs._subtree_acceptance_probability(rrs5))
    print ('Acceptance Prob 2-3: %f' % rrs2._subtree_acceptance_probability(rrs3))
    print ('Acceptance Prob 2-4: %f' % rrs2._subtree_acceptance_probability(rrs4))
    print ('Acceptance Prob 2-5: %f' % rrs2._subtree_acceptance_probability(rrs5))
    print ('Acceptance Prob 3-4: %f' % rrs3._subtree_acceptance_probability(rrs4))
    print ('Acceptance Prob 3-5: %f' % rrs3._refine_part_position_acceptance_probability(rrs5, 0))
    print ('Acceptance Prob 4-5: %f' % rrs4._subtree_acceptance_probability(rrs5))
    print((rrs.prior*rrs.likelihood)/(rrs2.prior*rrs2.likelihood))
#     forward_model._view(rrs)
#     forward_model._view(rrs2)
#     forward_model._view(rrs3)
#     forward_model._view(rrs4)
#     forward_model._view(rrs5)