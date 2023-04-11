import numpy as np

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
        pass

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
    def __init__(self, num_children: int):
        super().__init__(num_children)

    def compute(self):
        return self.compute(self.children[0]) + self.compute(self.children[1])
    
class Sin(Node):
    def __init__(self, num_children: int):
        super().__init__(num_children)

    def compute(self):
        return np.sin(self.compute(self.children[0]))