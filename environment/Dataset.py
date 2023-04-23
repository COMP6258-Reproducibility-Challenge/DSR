import numpy as np

from Expr import Expr
import torch
from NodeLibrary import Library
from nodes import Node



class Dataset():
    def __init__(self, target_expr: Expr, numpoints=100):
        # Or np.uniform? Whatever the paper says
        self.X = torch.linspace(-1, 1, numpoints)
        self.Y = torch.linspace(-1, 1, numpoints)
        self.z = target_expr.expr_func(self.X, self.Y)

    def NRMSELoss(self, yhat, y):
        return (1/len(y)) * torch.sqrt(torch.mean((yhat-y)**2))

    def reward(self, expr: Expr):
        """
        Params:
            expr: Expr
                The expression to test

        Returns:
            float which is the squashed nrmse (reward) of the proposed expression on the dataset
        """

        yhat = expr.expr_func(self.X, self.Y)

        return 1/(1 + self.NRMSELoss(yhat, self.z))

