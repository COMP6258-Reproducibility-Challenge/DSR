import numpy as np

from .Expr import Expr
import torch
from .NodeLibrary import Library
from .nodes import Node

class Dataset():
    def __init__(self, target_expr: Expr, numpoints=100, lb=-1, ub=1):
        # Or np.uniform? Whatever the paper says
        # self.X = torch.linspace(-1, 1, numpoints)
        # self.Y = torch.linspace(-1, 1, numpoints)
        self.X = torch.distributions.Uniform(low=lb, high=ub).sample((numpoints,))
        self.Y = torch.distributions.Uniform(low=lb, high=ub).sample((numpoints,))
        self.z = target_expr.expr_func(self.X, self.Y)
        self.normalising_const = torch.std(self.z)

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

    def grad_reward(self, expr: Expr):
        """
        Params:
            expr: Expr
                The expression to test

        Returns:
            float which is the squashed nrmse (reward) of the proposed expression on the dataset which has gradients enabled
        """

        yhat = expr.expr_func(self.X, self.Y)

        return 1/(1 + self.NRMSELoss(yhat, self.z))
