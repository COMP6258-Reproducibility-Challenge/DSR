import numpy as np

from Expr import Expr

class Dataset():
    def __init__(self, expr: Expr, numpoints=100):
        # Or np.uniform? Whatever the paper says
        self.X = np.linspace(-1,1,numpoints)
        self.y = expr(self.X)
    
    def reward(self, expr: Expr):
        """
        Params:
            expr: Expr
                The expression to test
        
        Returns:
            float which is the squashed nrmse (reward) of the proposed expression on the dataset
        """
        return 0