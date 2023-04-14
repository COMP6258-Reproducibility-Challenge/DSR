import numpy as np

class Expr():
    """
    Represents an actual finished Expr from which stuff can be calculated
    """
    
    def __init__(self, node_list):
        """
        Takes in an expression tree and generates a callable function with a numpy array as input and output. The function
        and string representation generation should happen in __init__. The __call__ and __repr__ methods shouldnt need to
        be changed (unless we change the whole way this is done) as they depend only on expr_func and expr_repr being filled in.

        Params
            expr_tree: list[Node]
                A fully filled out expression tree in the form of a list of nodes for an in-order traversal of the tree.
        """
        self.expr_func = lambda x: x
        self.expr_repr = "x"

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