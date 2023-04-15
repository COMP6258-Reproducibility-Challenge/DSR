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
        self.node_list.append(node_to_add)

        #add the current node as a child of a node on the stack
        if len(self.node_list) > 1:
            self.stack.pop().add_child(node_to_add)

        #add the current node to the stack for every child space it has
        for _ in range(node_to_add.remaining_children()):
            self.stack.append(node_to_add)

    
    def valid_nodes_mask(self):
        """
        Returns a mask (list of boolean values) of the same shape as the library where a True reflects that 
        the corresponding node is valid and could be added next, and a False reflects that that node is an invalid
        addition. This will be needed to zero out the output probabilities of invalid nodes.
        """

        mask = [True for _ in range(len(self.library))]

        #make sure the tree is 4 or more nodes long
        if len(self.node_list) + len(self.stack) < 4 and len(self.stack) < 2:
            #TODO disallow adding terminals
            pass

        #make sure the tree is 30 or less nodes long
        else:
            if len(self.node_list) + len(self.stack) > 28:
                #TODO disallow adding binary operators
                pass
            if len(self.node_list) + len(self.stack) > 29:
                #TODO disallow adding unary operators
                pass
        pass