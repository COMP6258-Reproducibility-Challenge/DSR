from .NodeLibrary import Library
import numpy as np
import torch
from scipy.optimize import minimize


class Expr():

    def __init__(self, library: Library, node_list=[]):
        """
        Params
            library: Library
                Library of available nodes
            node_list: list[Node]
                In-order listing of nodes in the expression tree (normally starts off empty)
        """
        self.node_list = node_list
        self.library = library

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
        return self.node_list[0].stringify()

    def expr_func(self, x: torch.Tensor, y: torch.Tensor):
        for node in self.node_list:
            if type(node).__name__ == "X":
                node.set_value(x)
            if type(node).__name__ == "Y":
                node.set_value(y)
        return self.node_list[0].compute()
    
    def func_to_optimize(self,consts,dataset):
        self.set_const(consts)
        reward = dataset.reward(self)
        # Constant optimizer minimizes the objective function
        return -reward
    
    def jac(self,consts,dataset):
        vals = self.set_const(consts)
        reward = -dataset.grad_reward(self)
        reward.backward(retain_graph=True)
        grads = []
        for const in vals:
            grads.append(const.grad.item())
        return grads
    
    def set_const(self,consts):
        vals = []
        count = 0
        for node in self.node_list:
            if node.__class__.__name__ == "Const":
                node.set_value(consts[count])
                vals.append(node.compute())
                count+=1
        return vals
    
    def optimise_consts(self, dataset):
        count = 0
        for node in self.node_list:
            if node.__class__.__name__ == "Const":
                count+=1
        x0 = np.ones(count)
        if count != 0 :
            opt = minimize(self.func_to_optimize,x0,args=(dataset),jac=self.jac,method='L-BFGS-B',options = {"gtol" : 1e-3})
            new_const = opt["x"]
            self.set_const(new_const)
        return