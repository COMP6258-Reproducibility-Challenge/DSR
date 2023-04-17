from Expr import Expr
from NodeLibrary import Library
import numpy as np
import torch


class Expr():

    def __init__(self, library: Library, node_list=[]):
        """
        Params
            library: Library
                Library of available nodes
            node_list: list[Node]
                In-order listing of nodes in the expression tree (normally starts off empty)
        """
        self.stack = []
        self.node_list = node_list
        self.library = library
        pass

    def __call__(self, X: np.ndarray):
        """
        Params
            X: np.ndarray
                An n-dimensional numpy vector 

        Returns
            The output of the expression as a numpy array.
        """
        return self.expr_func(X)

    def __repr__(self):
        """
        Returns string representation of expression
        """
        return self.expr_repr

    def expr_func(self, x: torch.Tensor, y: torch.Tensor):
        for node in self.node_list:
            if type(node).__name__ == "X":
                node.set_value(x)
            if type(node).__name__ == "Y":
                node.set_value(y)
        return self.node_list[0].compute()

    def add_node(self, node_index: int):
        """
        Adds node to the tree.

        Params
            node_index: int
                The index of the node in the library
        """
        node_to_add = self.library.get_node(node_index)
        self.node_list.append(node_to_add)

        # add the current node as a child of a node on the stack
        if len(self.node_list) > 1:
            parent_node = self.stack.pop()
            parent_node.add_child(node_to_add)
            if parent_node.has_trig_ancestor():
                node_to_add.add_trig_ancestor()

        # add the current node to the stack for every child space it has
        for _ in range(node_to_add.remaining_children()):
            self.stack.append(node_to_add)

    def valid_nodes_mask(self):
        """
        Returns a mask (list of boolean values) of the same shape as the library where a True reflects that 
        the corresponding node is valid and could be added next, and a False reflects that that node is an invalid
        addition. This will be needed to zero out the output probabilities of invalid nodes.
        """
        mask = [True for _ in range(len(self.library))]

        # make sure the tree is 4 or more nodes long
        if len(self.node_list) + len(self.stack) < 4 and len(self.stack) < 2:
            mask = mask[:-3] + [False] * 3  # disallow terminals

        # make sure the tree is 30 or less nodes long
        else:
            if len(self.node_list) + len(self.stack) > 28:
                mask = [False] * 4 + mask[4:]  # disallow binary operators

            if len(self.node_list) + len(self.stack) > 29:
                mask = mask[:4] + [False] * 4 + \
                    mask[-3:]  # disallow unary operators

        next_parent = self.stack[-1]

        if next_parent.__class__.__name__ == "Log":
            mask[7] = False

        if next_parent.__class__.__name__ == "Exp":
            mask[8] = False

        if next_parent.num_children == 1:
            mask[-1] = False
        # c<-binary->(?) If binary operator has one constant child then cannot have another
        if next_parent.num_children == 2 and next_parent.remaining_children() == 1 and next_parent.children.__class__.__name__ == "Const":
            mask[-1] = False

        if next_parent.has_trig_ancestor():
            mask = mask[:4] + [False] * 2 + \
                mask[-5:]

        return mask
