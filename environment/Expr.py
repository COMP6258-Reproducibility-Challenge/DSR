from .NodeLibrary import Library
import numpy as np
import torch
from .Dataset import Dataset
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
        for i in range(self.node_list):
            node = self.node_list[i]
            if node.__class__.__name__ == "Const":
                node.set_value(consts[i])
        val = 1/(1 + dataset.NRMSELoss(self,self.z))
        return val
    
    
    def optimise_consts(self, dataset: Dataset):
        consts =[]
        values = []
        for node in self.node_list:
            if node.__class__.__name__ == "Const":
                consts.append(node)
                values.append(node.compute())
        opt = minimize(self.func_to_optimize,values,args=(dataset),method='BFGS',jac=self.func_to_optimize().backwards())
        for i in range(consts):
            consts[i].set_value(opt[i])
        return