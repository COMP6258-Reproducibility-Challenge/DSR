import numpy as np

from .Expr import Expr
import torch
from .NodeLibrary import Library
from .nodes import Node



class Dataset():
    """
    Defines the dataset that is used to calcuate the reward for the model.
    """
    def __init__(self, target_expr: Expr, numpoints=100, lb=-1, ub=1):
        self.X = torch.distributions.Uniform(low=lb, high=ub).sample((numpoints,))
        self.Y = torch.distributions.Uniform(low=lb, high=ub).sample((numpoints,))
        self.z = target_expr.expr_func(self.X, self.Y)
        self.normalising_const = torch.std(self.z)

    @torch.no_grad()
    def NRMSELoss(self, yhat, y):
        return (1/self.normalising_const) * torch.sqrt(torch.mean((yhat-y)**2))

    @torch.no_grad()
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

