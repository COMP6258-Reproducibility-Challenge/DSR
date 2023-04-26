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
        pass

    def get_size(self):
        return len(self.nodes)
    
    def get_node(self, node_index: int):
        return self.nodes[node_index]()

    def get_node_int(self, node: Node):
        for i in range(len(self.nodes)):
            lib_node = self.nodes[i]
            if node.__class__.__name__ == lib_node.__class__.__name__:
                return i
        return -1