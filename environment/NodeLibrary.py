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
        return self.nodes[node_index].duplicate()