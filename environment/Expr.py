from Expr import Expr
from NodeLibrary import Library


class Expr():

    def __init__(self, library: Library, node_list=[]):
        """
        Params
            library: Library
                Library of available nodes
            node_list: list[Node]
                In-order listing of nodes in the expression tree (normally starts off empty)
        """
        self.stack = []
        self.node_list = node_list
        self.library = library
        pass

    def build_equation(self) -> Expr:
        """
        Constructs callable equation from the ExprTree
        """
        return Expr(self.node_list)

    def add_node(self, node_index: int):
        """
        Adds node to the tree.

        Params
            node_index: int
                The index of the node in the library
        """
        node_to_add = self.library.get_node(node_index)

        if len(self.node_list) > 0:
            self.stack.pop().add_child(node_to_add)

        self.node_list.append(node_to_add)
        last_node = self.node_list[-1]
        if last_node.remaining_children() == 2:
            self.stack.append(last_node)
            self.stack.append(last_node)
        elif last_node.remaining_children() == 1:
            self.stack.append(last_node)

    def valid_nodes_mask(self):
        """
        Returns a mask (list of boolean values) of the same shape as the library where a True reflects that 
        the corresponding node is valid and could be added next, and a False reflects that that node is an invalid
        addition. This will be needed to zero out the output probabilities of invalid nodes.
        """
        # len(self.node_list) + len(self.stack) < 4 and len(self.stack) < 2 remove terminals
        pass
