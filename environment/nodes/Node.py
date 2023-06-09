import numpy as np
import torch
from abc import abstractmethod, ABC


class Node(ABC):
    """
    Abstract class for a node of an equation tree
    """

    def __init__(self, num_children):
        # num_children should be either 0,1,2
        self.num_children = num_children
        self.trig_ancestor = False
        # children are just a list of nodes (which may be empty if num_children == 0)
        self.children = []

    def add_child(self, child):
        """
        Params:
            child: Node
                The node to add as a child
        """
        if self.remaining_children() > 0:
            self.children.append(child)

    def remaining_children(self):
        """
        Returns the number of remaining empty spaces for children
        """
        return self.num_children-len(self.children)

    def has_trig_ancestor(self):
        """
        Returns a boolean that indicates whether the node has a trigonometric node as an ancestor
        """
        return self.trig_ancestor

    def add_trig_ancestor(self):
        """
        Flags the node as having a trigonometric ancestor 
        """
        self.trig_ancestor = True

    @abstractmethod
    def compute(self):
        """
        Recursively constructs a function that can be used to compute the output of the expression
        """
        pass

    @abstractmethod
    def stringify(self):
        """
        Recursively creates a string representation of the node
        """
        pass

    @abstractmethod
    def duplicate(self):
        """
        Creates a new instance of it's class
        """
        pass


class Add(Node):
    def __init__(self, num_children: int = 2):
        super().__init__(num_children)
        self.trig_ancestor = False

    def compute(self):
        return self.children[0].compute() + self.children[1].compute()

    def stringify(self):
        return "(" + self.children[0].stringify() + ")" + " + " + "(" + self.children[1].stringify() + ")"

    def duplicate(self):
        return Add()


class Sub(Node):
    def __init__(self, num_children: int = 2):
        super().__init__(num_children)
        self.trig_ancestor = False

    def compute(self):
        return self.children[0].compute() - self.children[1].compute()

    def stringify(self):
        return "(" + self.children[0].stringify() + ")" + " - " + "(" + self.children[1].stringify() + ")"

    def duplicate(self):
        return Sub()


class Mult(Node):
    def __init__(self, num_children: int = 2):
        super().__init__(num_children)
        self.trig_ancestor = False

    def compute(self):
        return self.children[0].compute() * self.children[1].compute()

    def stringify(self):
        return "(" + self.children[0].stringify() + ")" + " * " + "(" + self.children[1].stringify() + ")"

    def duplicate(self):
        return Mult()


class Div(Node):
    def __init__(self, num_children: int = 2):
        super().__init__(num_children)
        self.trig_ancestor = False

    def compute(self):
        return self.children[0].compute() / self.children[1].compute()

    def stringify(self):
        return "(" + self.children[0].stringify() + ")" + " / " + "(" + self.children[1].stringify() + ")"

    def duplicate(self):
        return Div()


class Sin(Node):
    def __init__(self, num_children: int = 1):
        super().__init__(num_children)
        self.trig_ancestor = True

    def compute(self):
        return torch.sin(self.children[0].compute())

    def stringify(self):
        return "sin(" + self.children[0].stringify() + ")"

    def duplicate(self):
        return Sin()


class Cos(Node):
    def __init__(self, num_children: int = 1):
        super().__init__(num_children)
        self.trig_ancestor = True

    def compute(self):
        return torch.cos(self.children[0].compute())

    def stringify(self):
        return "cos(" + self.children[0].stringify() + ")"

    def duplicate(self):
        return Cos()


class Log(Node):
    def __init__(self, num_children: int = 1):
        super().__init__(num_children)
        self.trig_ancestor = False

    def compute(self):
        return torch.log(self.children[0].compute())

    def stringify(self):
        return "log(" + self.children[0].stringify() + ")"

    def duplicate(self):
        return Log()


class Exp(Node):
    def __init__(self, num_children: int = 1):
        super().__init__(num_children)
        self.trig_ancestor = False

    def compute(self):
        return torch.exp(self.children[0].compute())

    def stringify(self):
        return "exp(" + self.children[0].stringify() + ")"

    def duplicate(self):
        return Exp()


class X(Node):
    def __init__(self, num_children: int = 0, value: torch.Tensor = torch.zeros(1)):
        super().__init__(num_children)
        self.value = value
        self.trig_ancestor = False

    def set_value(self, value: torch.Tensor):
        self.value = value

    def compute(self):
        return self.value

    def stringify(self):
        return "x"

    def duplicate(self):
        return X()


class Y(Node):
    def __init__(self, num_children: int = 0, value: torch.Tensor = torch.zeros(1)):
        super().__init__(num_children)
        self.value = value
        self.trig_ancestor = False

    def set_value(self, value: torch.Tensor):
        self.value = value

    def compute(self):
        return self.value

    def stringify(self):
        return "y"

    def duplicate(self):
        return Y()


class Const(Node):
    def __init__(self, num_children: int = 0, value: torch.Tensor = torch.ones(1,requires_grad=True)):
        super().__init__(num_children)
        self.value = value
        self.trig_ancestor = False

    def set_value(self, value: float):
        self.value = torch.Tensor([value])
        self.value.requires_grad = True

    def compute(self):
        return self.value

    def stringify(self):
        return str(self.value.tolist()[0])

    def duplicate(self):
        return Const()
