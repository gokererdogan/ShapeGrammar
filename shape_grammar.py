# -*- coding: utf-8 -*-
'''
Created on May 14, 2013
Analysis of Multisensory Representations
Base classes for shape grammar state and 
shape spatial model.
@author: gerdogan
Last Update: May 14, 2013
'''

from pcfg_tree import *
from vision_forward_model import *
from haptics_forward_model import *
import matplotlib.pyplot as pl

class SpatialModel:
    """
    Spatial Model Base Class for Shape Grammar
    Subclass this class to implement your own spatial
    model
    """
    def __init__(self):
        """
        Initialize spatial model
        If positions is not given, it is simply instantiated
        with an empty dictionary
        """
        pass
    def update(self, data):
        """
        Updates spatial model 
        """
        pass
    
    def propose(self, data):
        """
        Proposes a new spatial model based on current one.
        """    
        pass
    
    def probability(self):
        """
        Returns probability of model
        """
        pass
    
class ShapeGrammarState(PCFGTree):
    """
    State representation class for shape grammar
    Contains parse tree of representation in self.tree
    and spatial layout in self.spatial_model
    This class implements general functionality for any shape
    grammar, you should implement new classes that inherit this
    base class if you need more functionality.
    """
    # maximum depth allowed for a parse tree
    # get_random_tree returns trees that have depth =< MAX_DEPTH
    # reason for implementing this constraint is the fact that for
    # some grammars (where branching factor is high) the tree grows
    # without bound
    MAXIMUM_DEPTH = 3
    def __init__(self, grammar, forward_model, data, ll_params, spatial_model, initial_tree=None):
        """
        Initializes ShapeGrammarState
        grammar: Probabilistic context free shape grammar definition. PCFG instance.
        forward_model: Forward model(s) used in likelihood calculation
        ll_params: parameters for likelihood calculation
        spatial_model: Spatial model for the state. Initialized randomly if initial_tree is not
                        provided
        initial_tree (optional): Parse tree for the state. If not provided, state is initialized
                        randomly.
        """
        self.grammar = grammar
        self.spatial_model = spatial_model
        self.forward_model = forward_model
            
        if initial_tree is None:
            self.tree = self._get_random_tree(start=self.grammar.start_symbol, max_depth=self.MAXIMUM_DEPTH)
            # initialize spatial model
            self.spatial_model.update(self.tree, self.grammar)
        else:
            self.tree = initial_tree
            
            
        # IMPORTANT: we call base class init after we initialize spatial model, 
        # because prior, ll, deriv_prob calculation is done in init and
        # spatial_model should be available to calculate them
        PCFGTree.__init__(self, grammar, data, ll_params, self.tree)
        
    def propose_state(self):
        """
        Proposes a new state based on current state
        Uses base class PCFGTree's propose_state to generate new
        tree, then samples positions for new added parts
        """
        proposed_tree = PCFGTree.propose_tree(self)
        # get a new spatial model based on proposed tree
        proposed_spatial_model = self.spatial_model.propose(proposed_tree, self.grammar)
        # note that the signature of initializer is different for classes that subclass AoMRShapeState
        # first parameter grammar is omitted because subclasses should be grammar specific. 
        return self.__class__(self.forward_model, self.data, self.ll_params, proposed_spatial_model, proposed_tree)
            
        
    def _prior(self):
        """
        Prior probability for state
        Product of probability for tree and spatial layout
        """
        prior = PCFGTree._prior(self) * self.spatial_model.probability()
        
        return prior
                                                 
    def _likelihood(self):
        """
        Likelihood function
        Gets render from forward model and calculates distance between
        data and render.
        """
        data = self.data
        params = self.ll_params
        b = params['b']
        render = self.forward_model.render(self)
        distance = np.sum((render - data)**2) / float(render.size)
        return np.exp(-b*distance) 
        
    def acceptance_prob(self, proposal):
        """
        Acceptance probability of proposal state given current state
        Gets acceptance probability for tree from base class
        and augments it according to spatial layout probabilities
        NOTE: this acceptance probability may not be correct for your
        model, you should override this method in your subclass
        """
        acc_prob = PCFGTree.acceptance_prob(self, proposal) * ( proposal.spatial_model.probability() / self.spatial_model.probability())
        return acc_prob
         
    def convert_to_parts_positions(self):
        """
        Converts the state representation to parts and positions
        representation that can be given to forward model
        Override in super class
        """
        pass
    def __eq__(self, other):
        """
        Override in super class if checking for equality is needed
        Equality checking is a grammar, and spatial model specific 
        operation.
        """
        return False
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __repr__(self):
        """
        Override in super class for more informative string representations
        """
        return PCFGTree.__repr__(self)

    def __str__(self):
        """
        Override in super class for more informative string representations
        """
        return PCFGTree.__str__(self)
    
    def __getstate__(self):
        """
        Return data to be pickled. 
        ForwardModel cannot be pickled because it contains VTKObjects, that's
        why we remove it from data to be pickled
        """
        return dict((k,v) for k, v in self.__dict__.iteritems() if k is not 'forward_model')
    

