import numpy as np

class Equation():
    """
    Represents an actual finished equation from which stuff can be calculated
    """
    
    def __init__(self, equation_repr):
        """
        equation_repr is "something" which reprents the equation being passed in. In the __init__ function
        we should convert that representation into a numpy function that can be called (and it better run 
        *fast* or I will be upset). If for loops are needed then you will wanna look into Numba.
        Might make sense to have equation_repr just be an EquationTree
        """
        self.equation_repr = equation_repr

    def __call__(self, X: np.ndarray):
        """
        Params:
            X: np.ndarray
                An n-dimensional numpy vector 
        """
        return X

    def __repr__(self):
        """
        Returns string representation of equation
        """
        return "x"