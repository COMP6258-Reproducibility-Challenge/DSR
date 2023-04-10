from abc import abstractmethod, ABC

def Node(ABC):
    """
    This is a stub but might be useful for implementing equations
    This is an abstract class for a node of an equation tree

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
        pass