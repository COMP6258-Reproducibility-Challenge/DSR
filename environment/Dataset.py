import numpy as np

from Equation import Equation

class Dataset():
    def __init__(self, equation: Equation, numpoints=100):
        # Or np.uniform? Whatever the paper says
        self.X = np.linspace(-1,1,numpoints)
        self.y = equation(self.X)
    
    def error(self, test_equation: Equation):
        """
        Params:
            test_equation: Equation
                The equation to test the error of 
        
        Returns:
            float which is the squashed nrmse of the proposed equation on the dataset
        """
        return 0