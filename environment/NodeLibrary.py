from .nodes import Node

class Library():
    """
    Represents the set of operators that can be included in an equation
    """
    def __init__(self, nodes):
        """
        Params
            nodes: list[Node]
                The list of available nodes in the library
        """
        self.nodes = nodes
        self.names = [node().__class__.__name__ for node in self.nodes]

    def get_size(self):
        return len(self.nodes)
    
    def get_node(self, node_index: int):
        return self.nodes[node_index]()

    def get_node_int(self, node: Node):
        for i, name in enumerate(self.names):
            if node.__class__.__name__ == name:
                return i
        return -1