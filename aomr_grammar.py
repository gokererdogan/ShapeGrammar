# coding=utf-8
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
import itertools as it
import zss # for tree edit distance (https://github.com/timtadh/zhang-shasha)

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
                    self.voxels[node] = [0, 0, 0]
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
    
    def kernel_p(self, other):
        """
        Calculate P-kernel between this and other state
        Ref: Cristianini, N.; Shawe-Taylor, J., Kernel Methods for Pattern Analysis, 
            Chapter 12, p401
        This is called marginalisation kernel in the book.
        Note that P-kernel is actually calculated between observable
        variables in the generative model (in our case 2D images or joint
        angles), hence P-kernel can only be calculated from multiple samples
        Here we simply calculate a P-kernel value assuming that we only
        have one sample for each data item. We use these kernel values to
        calculate actual p-kernel values in data analysis code
        """
        # swap ll models
        self_data = self.data
        other_data = other.data
        self.data = other_data
        other.data = self_data
        # ll of self under other's model
        ll_self_uother = self._likelihood()
        # ll of other under self's model
        ll_other_uself = other._likelihood()
        # re-swap ll models
        self.data = self_data
        other.data = other_data
        # calculate kernel
        kern = self.prior * self.likelihood * ll_self_uother
        kern = kern + (other.prior * other.likelihood * ll_other_uself)
        return kern
    
    def kernel_subtree(self, other):
        """
        Calculate all subtrees kernel between this and other state
        We use the fact that subtree kernel is the sum of corooted 
        subtree kernel values between all pairs of 
        nodes in two trees.
        Note that we take into account the labels and voxel ids associated 
        with nodes when calculating kernel value. 
        """
        kern = 0
        t1 = self.tree
        t2 = other.tree
        for n1, n2 in it.product(t1.nodes, t2.nodes):
            kern = kern + self._kernel_count_corooted_subtrees(n1, other, n2)
        return kern
    
    def _kernel_count_corooted_subtrees(self, node_self, other, node_other):
        """
        Count corooted subtrees between subtree of self with root node_self
        and subtree of other with root node_other
        Ref: Cristianini, N.; Shawe-Taylor, J., Kernel Methods for Pattern Analysis, 
            Chapter 11, pp 385-387  
            This is basically code fragment 11.7 from the above book
        NOTE We require the labels for a node and its voxel ids (spatial model) to match.
        """
        # if the number of children do not match or their labels or voxel ids
        # do not match,  kernel value is 0
        # NOTE: this is different from the code in reference, we do not
        # want leaf nodes to have kernel value zero even if their labels are
        # the same, because this will make leaf labels irrelevant to kernel
        # calculation
        # also we pay attention to the spatial model. for two nodes to match
        # they should have the same voxel id too (if they have one)
        # note that in our spatial model y dimension has width 0, what y coordinate
        # is does not change the spatial location of node, that's why we ignore second
        # element (y coord) of voxel coordinates 
        n1 = self.tree[node_self]
        n2 = other.tree[node_other]
        voxel1 = self.spatial_model.voxels.get(node_self)
        # if the node does not have voxel coordinates, or it's root
        # in fact checking for root should not be necesssary since
        # all root nodes should have voxel coordinates 0,0,0 but 
        # it turns out due to a bug that may not be the case  
        if voxel1 is None or n1.bpointer is None:
            voxel1 = [0,0,0]
        voxel2 = other.spatial_model.voxels.get(node_other)
        if voxel2 is None or n2.bpointer is None: 
            voxel2 = [0,0,0]
        voxel1[1] = 0
        voxel2[1] = 0
        
        if len(n1.fpointer) != len(n2.fpointer) or \
            n1.tag.symbol != n2.tag.symbol or \
            n1.tag.rule != n2.tag.rule or \
            np.any(voxel1 != voxel2):
            return 0
        else:
            kern = 1
            # print('{0:s}-{1:s}'.format(n1.tag, n2.tag))
            # note that if you care about the order of children in a tree
            # this should be for node1, node2 in zip(n1.fpointer, n2.fpointer): 
            for node1, node2 in it.product(n1.fpointer, n2.fpointer):
                kern = kern * (self._kernel_count_corooted_subtrees(node1, other, node2) + 1)
            return kern

    def kernel_tree_edit_distance(self, other):
        """
        Calculate tree edit distance between this and other state
        Because tree edit distance is symmetric, we return the average of distances
        for both ways.
        We use the algorithm by 
            Kaizhong Zhang and Dennis Shasha. 
            Simple fast algorithms for the editing distance between trees 
            and related problems. SIAM Journal of Computing, 18:1245–1262, 1989
        We use the implementation from https://github.com/timtadh/zhang-shasha
        """
        t1 = self._kernel_tree_edit_distance_create_tree(self.tree.root)
        t2 = other._kernel_tree_edit_distance_create_tree(other.tree.root)
        return (zss.simple_distance(t1, t2) + zss.simple_distance(t2, t1)) / 2.0
    
    
    def _kernel_tree_edit_distance_create_tree(self, current_node):
        """
        Convert the state to tree format required by zss library for calculating
        tree edit distance
        """
        
        def get_label(s, n):
            """
            Returns the sort string and label for a tree node
            label is symbol in the node:voxel coordinates
            sort string also containt the rule number used at that node.
            sort string is used to sort the nodes of a tree into a canonical order.
            """
            voxel = s.spatial_model.voxels.get(n)
            # if there are no voxels associated with a node, or it is a root node
            if voxel is None or s.tree.nodes[n].bpointer is None:
                voxel = [0,0,0]
            voxel[1] = 0
            sort_str = s.tree.nodes[n].tag.symbol + str(s.tree.nodes[n].tag.rule)
            label = s.tree.nodes[n].tag.symbol
            voxel = ''.join([str(i) for i in voxel])
            sort_str = sort_str + ':' + voxel
            label = label + ':' + voxel
            return sort_str, label
        
        # get the sort strings for the children of this node
        child_sort_strs = [get_label(self, child)[0] for child in self.tree.nodes[current_node].fpointer]
        # get the permutation we need to get children into sorted order
        sort_indices = [i[0] for i in sorted(enumerate(child_sort_strs), key=lambda x: x[1])]
        # apply that permutation
        children = [self.tree.nodes[current_node].fpointer[i] for i in sort_indices]
        # call create_tree recursively on the children
        t = zss.Node(get_label(self, current_node)[1], children=[self._kernel_tree_edit_distance_create_tree(cn) for cn in children])
        return t



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
    


