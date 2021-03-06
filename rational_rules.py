# -*- coding: utf-8 -*-
'''
Created on May 9, 2013

@author: gerdogan
Last Update: May 13, 2013
'''
from pcfg_tree import *
from treelib import Node, Tree
import numpy as np
import scipy.special as sp
from copy import deepcopy

"""
Definition of RationalRules Probabilistic Context Free Grammar
"""
terminals = ['(', ')', 'v', '^', '0', '1', '1f0', '1f1', '2f0', '2f1', '3f0', '3f1', '4f0', '4f1']
nonterminals = ['D', 'C', 'P', 'F1', 'F2', 'F3', 'F4']
start_symbol = 'D'
rules = {'D' : [['(', 'C', ')', 'v', 'D'], ['0']], 'C' : [['P', '^', 'C'], ['1']], 
         'P' : [['F1'], ['F2'], ['F3'], ['F4']], 'F1' : [['1f0'], ['1f1']], 
         'F2' : [['2f0'], ['2f1']], 'F3' : [['3f0'], ['3f1']], 
         'F4' : [['4f0'], ['4f1']] }
prod_probabilities = {'D' : [.5, .5], 'C' : [.5, .5], 'P' : [.25, .25, .25, .25],
                      'F1' : [.5, .5], 'F2' : [.5, .5], 'F3' : [.5, .5], 'F4' : [.5, .5]}
# id of rules that produce only terminals for each nonterminal
# this is used for stopping tree from growing without bound and
# enforcing a depth limit
# 
terminating_rule_ids = {'D' : [1], 'C': [1], 'P' : [0, 1, 2, 3],
                        'F1' : [0, 1], 'F2' : [0, 1], 'F3' : [0, 1],
                        'F4' : [0, 1] }

rational_rules_pcfg = PCFG(terminals, nonterminals, start_symbol, rules, prod_probabilities, terminating_rule_ids)



class RationalRulesState(PCFGTree):
    """
    Representation class for RationalRules concept learning model
    Concepts are represented with boolean formulas generated by the
    above PCFG. RationalRulesState class contains the parse tree
    for the formula. Each node in the tree corresponds to a symbol
    in the grammar and children of each node are generated according
    to the production rules of the grammar. The symbol and the production
    rule used to derive its children are stored in tag variables in each
    node as a ParseNode object.
    Reference: Goodman, N. D., Tenenbaum, J. B., Feldman, J., & Griffiths, T. L. (2008).
    A rational analysis of rule-based concept learning. Cognitive science, 32(1), 108-54.
    """
    def __init__(self, grammar=None, data=None, ll_params=None, initial_tree=None):
        PCFGTree.__init__(self, grammar=rational_rules_pcfg, data=data, ll_params=ll_params, initial_tree=initial_tree)
    
    def _prior(self):
        # we multiply base class prior with 0.5 because 
        # we haven't included the nonterminal start symbol S
        # in our grammar that they use in the paper, this term
        # adds a beta(2)/beta(1) term to prior calculation
        # so we multiply by 0.5 to match the exact values in 
        # the paper
        return 0.5 * PCFGTree._prior(self)
    
    def _likelihood(self):
        """
        Likelihood function
        Calculated using Eq. 16 in ref.
        """
        data = self.data
        params = self.ll_params
        assert(data.shape[1]==5) #4 feature values, and 1 class label
        b = params['b']
        sample_count = data.shape[0]
        
        # get every conjunction clause (clauses within parentheses)
        formula = repr(self).replace("v0", '')
        clauses = []
        for s in formula.split('('):
            clauses.extend((s.split(')')))
        clauses = [c for c in clauses if c!='' and c!='v']
        
        #for each clause, evaluate the conjunction of terms
        clause_evals = np.ones((sample_count, len(clauses)), dtype=bool)
        for i, clause in enumerate(clauses):
            terms = clause.split('^')
            for term in terms:
                if len(term)>1:
                    feature_index = int(term[0]) - 1
                    value = int(term[2])
                    clause_evals[:, i] = np.logical_and(clause_evals[:, i], data[:, feature_index]==value)
        
        # results from each term is passed through or function to get estimated labels
        estimated_labels = np.any(clause_evals, 1)
        misclassified = np.sum(estimated_labels!=data[:,4])
        return np.exp(-b*misclassified) 
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        # actually this is not the right way to test for equality between two
        # sentences in rational_rules grammar, but for our purposes this does
        # the job without being too complicated to calculate
        return repr(self) == repr(other)
    
    def __ne__(self, other):
        return not self.__eq__(other)
            

if __name__ == '__main__':
    # Concept class to learn.
    # Table 1 in reference. First 4 values are feature values and fifth value is label
    data = np.array([[0, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1], [0, 0, 1, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0]])
    params = {'b': 6.0}
   
    #RANDOM TEST
    rrs = RationalRulesState(data=data, ll_params=params)
    print rrs
    print rrs.prior
    print rrs.derivation_prob
    print rrs.likelihood
    rrs.tree.show()
    rrs2, acc_prob = rrs.moves[0]()
    print rrs2
    print rrs2.prior
    print rrs2.derivation_prob
    print rrs2.likelihood
    rrs2.tree.show()
    
    print acc_prob
    
        
#     # TEST TREES
#     t1 = Tree()
#     t1.create_node(ParseNode('D', 0), identifier='D')
#     t1.create_node(ParseNode('(', ''), parent='D')
#     t1.create_node(ParseNode('C', 0), parent='D', identifier='C')
#     t1.create_node(ParseNode(')', ''), parent='D')
#     t1.create_node(ParseNode('v', ''), parent='D')
#     t1.create_node(ParseNode('D', 1), parent='D', identifier='D2')
#     t1.create_node(ParseNode('P', 0), parent='C', identifier='P')
#     t1.create_node(ParseNode('^', ''), parent='C')
#     t1.create_node(ParseNode('C', 1), parent='C', identifier='C2')
#     t1.create_node(ParseNode('0', ''), parent='D2')
#     t1.create_node(ParseNode('F1', 0), parent='P', identifier='F1')
#     t1.create_node(ParseNode('1', ''), parent='C2')
#     t1.create_node(ParseNode('1f0', ''), parent='F1')
#     
#     rrs = RationalRulesState(data=data, ll_params=params, initial_tree=t1)
#     print rrs
#     print rrs.prior
#     print rrs.likelihood
#     print rrs.prior * rrs.likelihood
#     
#     t2 = Tree()
#     
#     t2.create_node(ParseNode('D', 0), identifier='D')
#     t2.create_node(ParseNode('(', ''), parent='D')
#     t2.create_node(ParseNode('C', 0), parent='D', identifier='C')
#     t2.create_node(ParseNode(')', ''), parent='D')
#     t2.create_node(ParseNode('v', ''), parent='D')
#     t2.create_node(ParseNode('D', 1), parent='D', identifier='D2')
#     t2.create_node(ParseNode('0', ), parent='D2')
#     t2.create_node(ParseNode('P', 0), parent='C', identifier='P1')
#     t2.create_node(ParseNode('^', ''), parent='C')
#     t2.create_node(ParseNode('C', 0), parent='C', identifier='C2')
#     t2.create_node(ParseNode('F1', 0), parent='P1', identifier='F1')
#     t2.create_node(ParseNode('1f0', ''), parent='F1')
#     t2.create_node(ParseNode('P', 0), parent='C2', identifier='P2')
#     t2.create_node(ParseNode('^', ''), parent='C2')
#     t2.create_node(ParseNode('C', 1), parent='C2', identifier='C5')
#     t2.create_node(ParseNode('F3', 0), parent='P2', identifier='F3')
#     t2.create_node(ParseNode('3f0', ''), parent='F3')
#     t2.create_node(ParseNode('1', ''), parent='C5')
#     
#     rrs2 = RationalRulesState(data=data, ll_params=params, initial_tree=t2)
#     print rrs2
#     print rrs2.prior
#     print rrs2.likelihood
#     print rrs2.prior * rrs2.likelihood
# 
#     
#     print rrs._subtree_proposal_acceptance_probability(rrs2)
#     
#     t3 = Tree()
#     
#     t3.create_node(ParseNode('D', 0), identifier='D')
#     t3.create_node(ParseNode('(', ''), parent='D')
#     t3.create_node(ParseNode('C', 0), parent='D', identifier='C')
#     t3.create_node(ParseNode(')', ''), parent='D')
#     t3.create_node(ParseNode('v', ''), parent='D')
#     t3.create_node(ParseNode('D', 0), parent='D', identifier='D2')
#     t3.create_node(ParseNode('(', ''), parent='D2')
#     t3.create_node(ParseNode('C', 0), parent='D2', identifier='C3')
#     t3.create_node(ParseNode(')', ''), parent='D2')
#     t3.create_node(ParseNode('v', ''), parent='D2')
#     t3.create_node(ParseNode('D', 1), parent='D2', identifier='D3')
#     t3.create_node(ParseNode('0', ), parent='D3')
#     t3.create_node(ParseNode('P', 0), parent='C', identifier='P1')
#     t3.create_node(ParseNode('^', ''), parent='C')
#     t3.create_node(ParseNode('C', 0), parent='C', identifier='C2')
#     t3.create_node(ParseNode('F1', 0), parent='P1', identifier='F1')
#     t3.create_node(ParseNode('1f0', ''), parent='F1')
#     t3.create_node(ParseNode('P', 0), parent='C2', identifier='P2')
#     t3.create_node(ParseNode('^', ''), parent='C2')
#     t3.create_node(ParseNode('C', 1), parent='C2', identifier='C5')
#     t3.create_node(ParseNode('F3', 0), parent='P2', identifier='F3')
#     t3.create_node(ParseNode('3f0', ''), parent='F3')
#     t3.create_node(ParseNode('1', ''), parent='C5')
#     t3.create_node(ParseNode('P', 0), parent='C3', identifier='P3')
#     t3.create_node(ParseNode('^', ''), parent='C3')
#     t3.create_node(ParseNode('C', 0), parent='C3', identifier='C4')
#     t3.create_node(ParseNode('F2', 0), parent='P3', identifier='F2')
#     t3.create_node(ParseNode('2f0', ''), parent='F2')
#     t3.create_node(ParseNode('P', 0), parent='C4', identifier='P4')
#     t3.create_node(ParseNode('^', ''), parent='C4')
#     t3.create_node(ParseNode('C', 1), parent='C4', identifier='C6')
#     t3.create_node(ParseNode('F4', 0), parent='P4', identifier='F4')
#     t3.create_node(ParseNode('4f0', ''), parent='F4')
#     t3.create_node(ParseNode('1', ''), parent='C6')
#     
#     rrs3 = RationalRulesState(data=data, ll_params=params, initial_tree=t3)
#     print rrs3
#     print rrs3.prior
#     print rrs3.likelihood
#     print rrs3.prior * rrs3.likelihood
# 
#     
#     print rrs._subtree_proposal_acceptance_probability(rrs3)
#     print rrs2._subtree_proposal_acceptance_probability(rrs3)
#  