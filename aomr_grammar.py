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
from haptics_forward_model import HapticsForwardModel
from treelib import Tree
import numpy as np
from copy import deepcopy

"""
Definition of AoMR Probabilistic Context Free Shape Grammar
"""
terminals = ['Body', 'Front0', 'Front1', 'Bottom0', 'Bottom1', 'Top0', 'Top1', 'Ear0', 'Ear1']
nonterminals = ['S', 'P']
start_symbol = 'S'
rules = {'S' : [['S'], ['S', 'S'], ['S', 'S', 'S'], ['S', 'S', 'S', 'S'], 
                ['P'], ['P', 'S'], ['P', 'S', 'S'], ['P', 'S', 'S', 'S']],
        'P' : [['Body'], ['Front0'], ['Front1'], ['Bottom0'], ['Bottom1'], ['Top0'], ['Top1'], ['Ear0'], ['Ear1']], }
prod_probabilities = {'S' : [1/8.0, 1/8.0, 1/8.0, 1/8.0, 1/8.0, 1/8.0, 1/8.0, 1/8.0],
                      'P' : [1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0]}
# id of rules that produce only terminals for each nonterminal
# this is used for stopping tree from growing without bound and
# enforcing a depth limit
# 
terminating_rule_ids = {'S' : [4], 'P' : [0, 1, 2, 3, 4 ,5 ,6, 7, 8]}

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
        # get a random 3d vector with elements \in {-1,0,1} except [0,0,0]
        voxel = np.random.randint(-1, 2, 3)
        while (voxel == [0,0,0]).all() is True:
            voxel = np.random.randint(-1, 2, 3)
        return voxel
        
    def probability(self):
        """
        Returns probability of model
        """
        # each voxel id is uniform sample from 26 (3.3.3-1) possible values
        # for each item in voxels dictionary we have a voxel id sample as such
        return np.power(1.0 / 26.0, len(self.voxels))

    
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
                      self.change_part_proposal, self.add_remove_part_proposal]
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
            snode = self.tree[self.tree[node].bpointer].bpointer
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
                chosen_part_rule_id = np.random.choice(self.grammar.terminating_rule_ids['P'])
                chosen_part = self.grammar.rules['P'][chosen_part_rule_id][0]
                
                # add the new S branch, its child P node and its part
                new_snode = proposal_tree.create_node(tag=ParseNode('S', 4), parent=chosen_node_id)
                new_pnode = proposal_tree.create_node(tag=ParseNode('P', chosen_part_rule_id), parent=new_snode.identifier)
                proposal_tree.create_node(tag=ParseNode(chosen_part, ''), parent=new_pnode.identifier)
                
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
                # less than 4 children
                if proposal_tree[node].tag.symbol == 'S' and len(proposal_tree[node].fpointer) < 4:
                    suitable_nodes.append(node)
        elif move_type == 1: # remove move
            # find nodes that can be removed
            for node in proposal_tree.expand_tree(mode=Tree.WIDTH):
                # if node is not root, is an S node and directly goes to a part
                # and is not the only child of its parent
                if proposal_tree[node].bpointer is not None and \
                    proposal_tree[node].tag.symbol == 'S' and \
                    proposal_tree[node].tag.rule == 4 and \
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
        
    def add_remove_part_proposal(self):
        """
        Proposes a new state based on current state using add/remove part move
        First choose randomly if we are doing an add or remove move
        If add move: finds all S nodes with that does not have any child P node, and
            has less than 4 children.
            chooses one randomly, chooses a random part to append to it and adds a child
            P node to chosen S node
            This move essentially adds a new part to object at the level of chosen S node
            Add/remove branch move adds a new part and S node to a selected S node, in some
            sense adding a subpart to an existing part
        If remove move: finds all P nodes (except a P node that is the only P node in tree),
            chooses one randomly and removes it. We also need to remove its parent S node 
            if there are no children remaining after removing P node.
        """
        proposal_tree = deepcopy(self.tree)
        which_move = np.random.rand()
        move_type = -1
        if which_move < .5: # add move
            move_type = 0
            # find a suitable S node
            suitable_nodes = self._add_remove_part_get_suitable_nodes(proposal_tree, move_type)
            
            # if there are nodes to choose from, else do nothing
            if len(suitable_nodes) > 0:
                # choose one of the suitable nodes randomly
                chosen_node_id = np.random.choice(suitable_nodes)
                
                # sample a new part for the new P node
                chosen_part_rule_id = np.random.choice(self.grammar.terminating_rule_ids['P'])
                chosen_part = self.grammar.rules['P'][chosen_part_rule_id][0]
                
                # add the new P node and its part
                new_pnode = proposal_tree.create_node(tag=ParseNode('P', chosen_part_rule_id), parent=chosen_node_id)
                proposal_tree.create_node(tag=ParseNode(chosen_part, ''), parent=new_pnode.identifier)
                
                # change the production rule used in node's parent
                # NOTE: VERY GRAMMAR SPECIFIC CODE
                proposal_tree[chosen_node_id].tag.rule = proposal_tree[chosen_node_id].tag.rule + 5
        else: # remove move
            move_type = 1
            suitable_remove_nodes = self._add_remove_part_get_suitable_nodes(proposal_tree, move_type)
            
            # if we have nodes to choose from, else do nothing
            if len(suitable_remove_nodes) > 0:
                # choose one of the suitable nodes for removal
                chosen_node_id = np.random.choice(suitable_remove_nodes)
                # if this P node is the only child of its parent, remove parent S node
                # this is a little tricky since its parent can also be the single child of
                # its parent. (think of a long chain of S nodes). so we should go up until 
                # we find an S node with more than one child and remove its child S node
                node_to_remove = chosen_node_id
                while len(proposal_tree[proposal_tree[node_to_remove].bpointer].fpointer) == 1:
                    node_to_remove = proposal_tree[node_to_remove].bpointer
                    
                # update rule ids
                parent_node_id = proposal_tree[node_to_remove].bpointer
                if node_to_remove == chosen_node_id:
                    # update parent's rule id
                    proposal_tree[parent_node_id].tag.rule = proposal_tree[parent_node_id].tag.rule - 5
                else:
                    proposal_tree[parent_node_id].tag.rule = proposal_tree[parent_node_id].tag.rule - 1
                
                # remove nodes
                proposal_tree.remove_node(node_to_remove)
                
        # get a new spatial model based on proposed tree
        proposed_spatial_model = self.spatial_model.propose(proposal_tree, self.grammar)
        proposal = self.__class__(forward_model=self.forward_model, data=self.data, ll_params=self.ll_params, 
                                  spatial_model=proposed_spatial_model, initial_tree=proposal_tree)
        
        # get acceptance probability
        acc_prob = self._add_remove_part_acceptance_probability(proposal, move_type)
        return proposal, acc_prob    
    
    def _add_remove_part_get_suitable_nodes(self, proposal_tree, move_type):
        """
        Gets nodes that can be removed or added for add/remove part move depending
        on move_type parameter
        """
        suitable_nodes = []
        if move_type == 0: # add move
            for node in proposal_tree.expand_tree(mode=Tree.WIDTH):
                # if node is an S node and has
                # less than 4 children and none of them are a P node
                # NOTE: GRAMMAR SPECIFIC CODE
                if proposal_tree[node].tag.symbol == 'S' and proposal_tree[node].tag.rule < 3:
                    suitable_nodes.append(node)
        elif move_type == 1: # remove move
            # find nodes that can be removed
            for node in proposal_tree.expand_tree(mode=Tree.WIDTH):
                # if node is a P node, and is not the only P node in tree
                if proposal_tree[node].tag.symbol == 'P':
                    suitable_nodes.append(node)
            
            # if this is the only P node, we cannot remove it
            if len(suitable_nodes) == 1:
                suitable_nodes.pop()
        else:
            raise ValueError('move_type can only be 0 or 1')
        
        return suitable_nodes
    
    def _add_remove_part_acceptance_probability(self, proposal, move_type):
        """
        Acceptance probability for add/remove part move
        """
        # proposal probabilities
        q_sp_s = 1
        q_s_sp = 1
        if move_type == 0: # add move
            # number of branches we can remove in proposal
            removable_nodes = self._add_remove_part_get_suitable_nodes(proposal.tree, 1)
            # nodes to which we can add a branch in current state
            add_nodes = self._add_remove_part_get_suitable_nodes(self.tree, 0)
            if len(removable_nodes) > 0 and len(add_nodes) > 0:
                # q(S' -> S) or q(S|S')
                q_sp_s = (1.0 / len(removable_nodes))
                # q(S -> S') or q(S'|S)
                q_s_sp = (1.0 / len(add_nodes)) * (1.0 / 8.0)
        elif move_type == 1: # remove move
            # nodes to which we can add a branch in proposal
            add_nodes = self._add_remove_part_get_suitable_nodes(proposal.tree, 0)
            # nodes we can remove in current state
            removable_nodes = self._add_remove_part_get_suitable_nodes(self.tree, 1)
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
        proposal_tree[proposal_tree[chosen_node].bpointer].tag.rule = self.grammar.rules['P'].index([new_part])
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
    data = np.load('data/visual/15.npy')
    params = {'b': 1200.0}
    forward_model = VisionForwardModel(body_fixed=False)
    #h_forward_model = HapticsForwardModel(body_fixed=False)
    
    part1 = 'Bottom1'
    part2 = 'Front1'
    part3 = 'Top1'
    part4 = 'Ear0'
    # wrong parts for this object
    wpart1 = 'Bottom0'
    wpart2 = 'Front0'
    wpart3 = 'Top0'
    wpart4 = 'Ear1'
    
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
#     forward_model.save_image('random.png', rs)

#     
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
    
    # correct tree
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
    rrs4.tree.show()
    
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
#     forward_model._view(rrs)
