'''
Analysis of Multisensory Representations 
Simple Shape Grammar Implementation
This a very simple grammar used for testing 
purposes.
Created on Jun 4, 2013

@author: goker
'''

from pcfg_tree import *
from vision_forward_model import VisionForwardModel, actual_positions
from shape_grammar import ShapeGrammarState, SpatialModel

"""
Definition of Simple AoMR Probabilistic Context Free Shape Grammar
This is a very simple grammar that generates binary trees. Part 
locations are independent of tree structure. They are sampled from 
a uniform distribution over actual (veridical) part locations.
"""
terminals = ['Front0', 'Front1', 'Bottom0', 'Bottom1', 'Top0', 'Top1', 'Ear0', 'Ear1', 'Null']
nonterminals = ['S', 'P']
start_symbol = 'S'
rules = {'S' : [['P', 'S'], ['Null']], 
         'P' : [['Front0'], ['Front1'], ['Bottom0'], ['Bottom1'], ['Top0'], ['Top1'], ['Ear0'], ['Ear1']] }
prod_probabilities = {'S' : [.5, .5], 'P' : [.125, .125, .125, .125, .125, .125, .125, .125]}
terminating_rule_ids = {'S' : [1], 'P' : [0, 1, 2, 3, 4, 5, 6, 7]}

aomr_simple_shape_pcfg = PCFG(terminals, nonterminals, start_symbol, rules, prod_probabilities, terminating_rule_ids)


class AoMRSimpleSpatialModel(SpatialModel):
    """
    Spatial Model Class for AoMR Simple Shape Grammar
    Simply holds possible positions for parts
    Keys for positions dictionary are tree node ids
    """
    # Spatial Model Possible Locations
    possible_positions = [(0.0, 0.0, -0.0228), (0.05346666666666666, 0.0, 0.0076), (-0.017822222222222222, 0.0, 0.0304), (-0.017822222222222222, 0.0, 0.0228)]
    # probability for each position (uniform)
    prob = .25
    
    def __init__(self, positions=None):
        """
        Initialize spatial model
        If positions is not given, it is simply instantiated
        with an empty dictionary
        """
        if positions is None:
            self.positions = {}
        else:
            self.positions = positions
    # -------------------------------------------
    # TO-DO: I don't like passing tree and grammar to below methods, grammar
    # should somehow be accessible to this class already. Think about this later.
    # -------------------------------------------
    def update(self, tree, grammar):
        """
        Updates spatial model, removes nodes that are not in
        nodes parameter from positions dictionary and samples
        positions for newly added nodes 
        """
        new_nodes = [node for node in tree.expand_tree(mode=Tree.WIDTH) 
                               if tree[node].tag.symbol in grammar.terminals and 
                                  tree[node].tag.symbol is not 'Null']
        old_nodes = self.positions.keys()
        removed_nodes = [node for node in old_nodes if node not in new_nodes]
        added_nodes = [node for node in new_nodes if node not in old_nodes]
        
        for n in removed_nodes:
            del self.positions[n]
        for n in added_nodes:
            self.positions[n] = self._get_random_position()
    
    def propose(self, tree, grammar):
        """
        Proposes a new spatial model based on current one.
        Creates a new spatial model with current positions,
        updates it, and returns it
        """    
        positions_copy = deepcopy(self.positions)
        proposed_spatial_model = AoMRSimpleSpatialModel(positions_copy)
        proposed_spatial_model.update(tree, grammar)
        return proposed_spatial_model
    
    def _get_random_position(self):
        """
        Returns a random position
        """
        ix = np.random.choice(len(self.possible_positions))
        return self.possible_positions[ix]
    
    def probability(self):
        """
        Returns probability of model
        """
        return np.power(self.prob, len(self.positions))

    
class AoMRSimpleShapeState(ShapeGrammarState):
    """
    AoMR shape state class for simple AoMR grammar and spatial model
    """
    
    def __init__(self, forward_model=None, data=None, ll_params=None, spatial_model=None, initial_tree=None):
        """
        Constructor for AoMRSimpleShapeState
        Note that the first parameter ``grammar`` of base class AoMRShapeState is removed because 
        this class is a grammar specific implementation
        """
        self.MAXIMUM_DEPTH = 5
        ShapeGrammarState.__init__(self, grammar=aomr_simple_shape_pcfg, forward_model=forward_model, 
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
        parts = []
        positions = []
        for node, pos in self.spatial_model.positions.iteritems():
            parts.append(self.tree[node].tag.symbol)
            positions.append(pos)
        return parts, positions
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        
        self_parts_pos = []
        other_parts_pos = []
        for node, pos in self.spatial_model.positions.iteritems():
            self_parts_pos.append([self.tree[node].tag.symbol, pos])
            
        for node, pos in other.spatial_model.positions.iteritems():
            other_parts_pos.append([other.tree[node].tag.symbol, pos])
        
        self_parts_pos.sort()
        other_parts_pos.sort()
        return self_parts_pos == other_parts_pos
    
    def __neq__(self, other):
        return not self.__eq__(other)
    
    def __repr__(self):
        return "".join(self.tree[node].tag.symbol+repr(self.spatial_model.positions[node]) for node in self.tree.expand_tree(mode=Tree.DEPTH) 
                       if self.tree[node].tag.symbol in self.grammar.terminals and
                          self.tree[node].tag.symbol is not 'Null')
        
    def __str__(self):
        return "".join(self.tree[node].tag.symbol+repr(self.spatial_model.positions[node]) for node in self.tree.expand_tree(mode=Tree.DEPTH) 
                       if self.tree[node].tag.symbol in self.grammar.terminals and
                          self.tree[node].tag.symbol is not 'Null')
        

if __name__ == '__main__':
    # best b values: for visual 750, for haptics 
        # object 1: 5000
        # object 2: 9000
        # object 3: 5000 (maybe lower)
        # object 4: 6000
    data = np.load('data/visual/1.npy')
    params = {'b': 750.0}
    forward_model = VisionForwardModel()
    #forward_model = HapticsForwardModel()
    
    # TEST TREES: We look at the prior, likelihood and acceptance probabilities for
    # empty tree, correct configuration (4 parts in correct positions) and tree with 
    # 1 part (ear) removed. Our purpose is to understand the b value we should set
    # to make sure correct configuration has the highest posterior.
      
    # Tree with no parts
    t1 = Tree()
    t1.create_node(ParseNode('S', 1), identifier='S')
    t1.create_node(ParseNode('Null', ''), parent='S')
     
    spatial_model1 = AoMRSimpleSpatialModel()
     
    rrs = AoMRSimpleShapeState(forward_model=forward_model, data=data, 
                               ll_params=params, spatial_model=spatial_model1, initial_tree=t1)
     
    print rrs
    print ('Prior: %g' % rrs.prior)
    print ('Likelihood: %g' % rrs.likelihood)
    print ('Posterior: %g' % (rrs.prior*rrs.likelihood))
    rrs.tree.show()
     
    # correct tree
    t2 = Tree()
    t2.create_node(ParseNode('S', 0), identifier='S')
    t2.create_node(ParseNode('P', 0), parent='S', identifier='P1')
    t2.create_node(ParseNode('S', 0), parent='S', identifier='S1')
    t2.create_node(ParseNode('Front0', ''), parent='P1', identifier='F0')
    t2.create_node(ParseNode('P', 2), parent='S1', identifier='P2')
    t2.create_node(ParseNode('S', 0), parent='S1', identifier='S2')
    t2.create_node(ParseNode('Bottom0', ''), parent='P2', identifier='B0')
    t2.create_node(ParseNode('P', 4), parent='S2', identifier='P3')
    t2.create_node(ParseNode('S', 0), parent='S2', identifier='S3')
    
    t2.create_node(ParseNode('Top0', ''), parent='P3', identifier='T0')
    t2.create_node(ParseNode('P', 6), parent='S3', identifier='P4')
    t2.create_node(ParseNode('S', 1), parent='S3', identifier='S4')
    t2.create_node(ParseNode('Ear0', ''), parent='P4', identifier='E0')
    t2.create_node(ParseNode('Null', ''), parent='S4')
    
    positions2 = {'F0' : actual_positions['Front0'], 
                 'B0' : actual_positions['Bottom0'], 
                 'T0' : actual_positions['Top0'],
                 'E0' : actual_positions['Ear0'], }
    
    spatial_model2 = AoMRSimpleSpatialModel(positions2)
    
    rrs2 = AoMRSimpleShapeState(forward_model=forward_model, data=data, 
                               ll_params=params, spatial_model=spatial_model2, initial_tree=t2)
    
    print rrs2
    print ('Prior: %g' % rrs2.prior)
    print ('Likelihood: %g' % rrs2.likelihood)
    print ('Posterior: %g' % (rrs2.prior*rrs2.likelihood))
    rrs2.tree.show()
    
    print ('Acceptance Prob 1-2: %f' % rrs._subtree_acceptance_probability(rrs2))
     
    # tree with 1 part missing
    t3 = Tree()
    t3.create_node(ParseNode('S', 0), identifier='S')
    t3.create_node(ParseNode('P', 0), parent='S', identifier='P1')
    t3.create_node(ParseNode('S', 0), parent='S', identifier='S1')
    t3.create_node(ParseNode('Front0', ''), parent='P1', identifier='F0')
    t3.create_node(ParseNode('P', 2), parent='S1', identifier='P2')
    t3.create_node(ParseNode('S', 0), parent='S1', identifier='S2')
    t3.create_node(ParseNode('Bottom0', ''), parent='P2', identifier='B0')
    t3.create_node(ParseNode('P', 4), parent='S2', identifier='P3')
    t3.create_node(ParseNode('S', 1), parent='S2', identifier='S3')
    t3.create_node(ParseNode('Ear0', ''), parent='P3', identifier='E0')
    t3.create_node(ParseNode('Null', ''), parent='S3')
     
    positions3 = {'F0' : actual_positions['Front0'], 
                 'B0' : actual_positions['Bottom0'], 
                 'E0' : actual_positions['Ear0'],}
     
    spatial_model3 = AoMRSimpleSpatialModel(positions3)
     
    rrs3 = AoMRSimpleShapeState(forward_model=forward_model, data=data, 
                               ll_params=params, spatial_model=spatial_model3, initial_tree=t3)
    print rrs3
    print ('Prior: %g' % rrs3.prior)
    print ('Likelihood: %g' % rrs3.likelihood)
    print ('Posterior: %g' % (rrs3.prior*rrs3.likelihood))
    rrs3.tree.show()
     
    print ('Acceptance Prob 1-3: %f' % rrs._subtree_acceptance_probability(rrs3))
    print ('Acceptance Prob 2-3: %f' % rrs2._subtree_acceptance_probability(rrs3))
     
    # tree with only bottom and ear
    t4 = Tree()
    t4.create_node(ParseNode('S', 0), identifier='S')
    t4.create_node(ParseNode('P', 0), parent='S', identifier='P1')
    t4.create_node(ParseNode('S', 0), parent='S', identifier='S1')
    t4.create_node(ParseNode('Bottom0', ''), parent='P1', identifier='B0')
    t4.create_node(ParseNode('P', 2), parent='S1', identifier='P2')
    t4.create_node(ParseNode('S', 1), parent='S1', identifier='S2')
    t4.create_node(ParseNode('Ear0', ''), parent='P2', identifier='E0')
    t4.create_node(ParseNode('Null', ''), parent='S2')
     
    positions4 = {'B0' : actual_positions['Bottom0'], 
                 'E0' : actual_positions['Ear0'],}
     
    spatial_model4 = AoMRSimpleSpatialModel(positions4)
     
    rrs4 = AoMRSimpleShapeState(forward_model=forward_model, data=data, 
                               ll_params=params, spatial_model=spatial_model4, initial_tree=t4)
    print rrs4
    print ('Prior: %g' % rrs4.prior)
    print ('Likelihood: %g' % rrs4.likelihood)
    print ('Posterior: %g' % (rrs4.prior*rrs4.likelihood))
    rrs4.tree.show()
     
    print ('Acceptance Prob 1-4: %f' % rrs._subtree_acceptance_probability(rrs4))
    print ('Acceptance Prob 2-4: %f' % rrs2._subtree_acceptance_probability(rrs4))
    print ('Acceptance Prob 3-4: %f' % rrs3._subtree_acceptance_probability(rrs4))
    
    forward_model._view(rrs2)
