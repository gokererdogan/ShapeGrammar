'''
Created on Jul 8, 2013
Development code for tree kernels and p-kernel
@author: goker
'''

from treelib import Tree
from pcfg_tree import ParseNode
import itertools as it

def kernel_subtree(t1, t2):
    """
    Calculate all subtrees kernel. We use the fact that subtree kernel
    is the sum of corooted subtree kernel values between all pairs of 
    nodes in two trees.
    """
    kern = 0
    for n1, n2 in it.product(t1.nodes, t2.nodes):
        kern = kern + _count_corooted_subtrees(t1, n1, t2, n2)
    return kern

def kernel_corooted_subtree(t1, t2):
    """
    Calculate corooted subtree kernel between trees t1 and t2
    Ref: Cristianini, N.; Shawe-Taylor, J., Kernel Methods for Pattern Analysis, 
        Chapter 11, pp 385-387  
    NOTE this method is for labelled trees. we require the labels to match.
    """
    return _count_corooted_subtrees(t1, t1.root, t2, t2.root)

def _count_corooted_subtrees(t1, root1, t2, root2):
    """
    Count corooted subtrees between subtree of t1 with root root1
    and subtree of t2 with root root2
    Ref: Cristianini, N.; Shawe-Taylor, J., Kernel Methods for Pattern Analysis, 
        Chapter 11, pp 385-387  
        This is basically code fragment 11.7 from the above book
    NOTE this method is for labelled trees. we require the labels to match.
    """
    # if the number of children do not match or their labels do not match
    # kernel value is 0
    # NOTE: this is different from the code in reference, we do not
    # want leaf nodes to have kernel value zero even if their labels are
    # the same, because this will make leaf labels irrelevant to kernel
    # calculation
    if len(t1[root1].fpointer) != len(t2[root2].fpointer) or \
        t1[root1].tag.symbol != t2[root2].tag.symbol:
        return 0
    else:
        kern = 1
        for node1, node2 in zip(t1[root1].fpointer, t2[root2].fpointer):
            kern = kern * (_count_corooted_subtrees(t1, node1, t2, node2) + 1)
        return kern

if __name__ == '__main__':
        
    part1 = 'Bottom1'
    part2 = 'Front1'
    part3 = 'Top1'
    part4 = 'Ear0'
    # wrong parts for this object
    wpart1 = 'Bottom0'
    wpart2 = 'Front0'
    wpart3 = 'Top0'
    wpart4 = 'Ear1'
    
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

    
    print('Corooted subtree kernel: {0:d}'.format(kernel_corooted_subtree(t2, t2)))
    print('Corooted subtree kernel: {0:d}'.format(kernel_corooted_subtree(t2, t3)))
    print('Corooted subtree kernel: {0:d}'.format(kernel_corooted_subtree(t3, t2)))
    print('Corooted subtree kernel: {0:d}'.format(kernel_corooted_subtree(t2, t6)))
    print('Corooted subtree kernel: {0:d}'.format(kernel_corooted_subtree(t6, t2)))
    print('All subtree kernel: {0:d}'.format(kernel_subtree(t2, t2)))
    print('All subtree kernel: {0:d}'.format(kernel_subtree(t2, t3)))
    print('All subtree kernel: {0:d}'.format(kernel_subtree(t3, t2)))
    print('All subtree kernel: {0:d}'.format(kernel_subtree(t2, t6)))
    print('All subtree kernel: {0:d}'.format(kernel_subtree(t6, t2)))