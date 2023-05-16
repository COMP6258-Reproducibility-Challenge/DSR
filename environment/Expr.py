from .NodeLibrary import Library
import numpy as np
import torch



class Expr():
    """
    Defines the expression to allow string representation and calculating the output
    """
    def __init__(self, library: Library, node_list=[]):
        """
        Params:
            library: Library
                Library of available nodes
            node_list: list[Node]
                In-order listing of nodes in the expression tree (normally starts off empty)
        """
        self.node_list = node_list
        self.library = library

    def __call__(self, X: np.ndarray):
        """
        Params:
            X: np.ndarray
                An n-dimensional numpy vector 

        Returns
            The output of the expression as a numpy array.
        """
        return self.expr_func(X)

    def __repr__(self):
        """
        Returns: 
            The string representation of expression
        """
        return self.node_list[0].stringify()

    def expr_func(self, x: torch.Tensor, y: torch.Tensor):
        for node in self.node_list:
            if type(node).__name__ == "X":
                node.set_value(x)
            if type(node).__name__ == "Y":
                node.set_value(y)
        return self.node_list[0].compute()
    