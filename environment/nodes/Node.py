import numpy as np
import torch
from abc import abstractmethod, ABC


def Node(ABC):
    """
    Abstract class for a node of an equation tree
    """

    def __init__(self, num_children):
        # num_children should be either 0,1,2
        self.num_children = num_children
        # children are just a list of nodes (which may be empty if num_children == 0)
        self.children = [None for _ in range(num_children)]

    # Need some logic here
    def add_child(self, child: Node):
        if self.children[0] == None:
            self.children[0] = child
        else:  # Assuming valid input where no additional children is added over max
            self.children[1] = child

    @abstractmethod
    def compute(self):
        """
        So this kinda lends itself to recursive construction of functions but not sure how that will affect speed.
        Maybe it wont actually matter too much? Will need to do some profiling.
        Either way we will just implement it in whatever way makes sense at first and optimise later.
        """
        pass


# Just got a couple of examples here. Not sure if this is the best way to do things so I am open to other
# suggestions on how to structure this.
class Add(Node):
    def __init__(self, num_children: int = 2):
        super().__init__(num_children)

    def compute(self):
        return self.compute(self.children[0]) + self.compute(self.children[1])


class Sub(Node):
    def __init__(self, num_children: int = 2):
        super().__init__(num_children)

    def compute(self):
        return self.compute(self.children[0]) - self.compute(self.children[1])


class Div(Node):
    def __init__(self, num_children: int = 2):
        super().__init__(num_children)

    def compute(self):
        return self.compute(self.children[0]) / self.compute(self.children[1])


class Sin(Node):
    def __init__(self, num_children: int = 1):
        super().__init__(num_children)

    def compute(self):
        return np.sin(self.compute(self.children[0]))


class Cos(Node):
    def __init__(self, num_children: int = 1):
        super().__init__(num_children)

    def compute(self):
        return np.cos(self.compute(self.children[0]))


class Log(Node):
    def __init__(self, num_children: int = 1):
        super().__init__(num_children)

    def compute(self):
        return np.log(self.compute(self.children[0]))


class Exp(Node):
    def __init__(self, num_children: int = 1):
        super().__init__(num_children)

    def compute(self):
        return np.exp(self.compute(self.children[0]))


class X(Node):
    def __init__(self, num_children: int = 0, value: torch.Tensor = torch.zeros(1)):
        super().__init__(num_children)
        self.value = value

    def compute(self):
        return self.value


class Y(Node):
    def __init__(self, num_children: int = 0, value: torch.Tensor = torch.zeros(1)):
        super().__init__(num_children)
        self.value = value

    def compute(self):
        return self.value


class Const(Node):
    def __init__(self, num_children: int = 0, value: float = 0):
        super().__init__(num_children)
        self.value = value

    def compute(self):
        return self.value
