from .Expr import Expr
from .NodeLibrary import Library

class ExprTree():

    def __init__(self, library: Library, node_list=[]):
        """
        Params
            library: Library
                Library of available nodes
            node_list: list[Node]
                In-order listing of nodes in the expression tree (normally starts off empty)
        """
        self.node_list = node_list
        self.library = library
        self.stack = []

    def build_expression(self) -> Expr:
        """
        Constructs callable equation from the ExprTree
        """
        return Expr(self.library, self.node_list)

    def get_parent_node(self):
        if len(self.stack) != 0:
            return self.stack[-1]  
        return None

    def get_sibling_node(self):
        if len(self.stack) == 0:
            return None
        parent = self.stack[-1]
        if parent.num_children == 2 and len(parent.children) == 1:
            return parent.children[0]
        return None

    def add_node(self, node_index: int):
        """
        Adds node to the tree.

        Params
            node_index: int
                The index of the node in the library
        """
        node_to_add = self.library.get_node(node_index)
        self.node_list.append(node_to_add)

        # add the current node as a child of a node on the stack
        if len(self.node_list) > 1:
            parent_node = self.stack.pop()
            parent_node.add_child(node_to_add)
            if parent_node.has_trig_ancestor():
                node_to_add.add_trig_ancestor()

        # add the current node to the stack for every child space it has
        for _ in range(node_to_add.remaining_children()):
            self.stack.append(node_to_add)

    def valid_nodes_mask(self):
        """
        Returns a mask (list of boolean values) of the same shape as the library where a True reflects that 
        the corresponding node is valid and could be added next, and a False reflects that that node is an invalid
        addition. This will be needed to zero out the output probabilities of invalid nodes.
        """
        mask = [True for _ in range(self.library.get_size())]
        if len(self.node_list) == 0:
            return mask

        # make sure the tree is 4 or more nodes long
        if len(self.node_list) + len(self.stack) < 4 and len(self.stack) < 2:
            # disallow terminals
            for i in range(self.library.get_size()):
                if self.library.get_node(i).num_children == 0:
                    mask[i] = False

        # make sure the tree is 30 or less nodes long
        else:
            if len(self.node_list) + len(self.stack) > 28:
                # disallow binary operators
                for i in range(self.library.get_size()):
                    if self.library.get_node(i).num_children == 2:
                        mask[i] = False

            if len(self.node_list) + len(self.stack) > 29:
                # disallow unary operators
                for i in range(self.library.get_size()):
                    if self.library.get_node(i).num_children == 1:
                        mask[i] = False

        next_parent = self.stack[-1]

        if next_parent.__class__.__name__ == "Log":
            for i in range(self.library.get_size()):
                if self.library.get_node(i).__class__.__name__ == "Exp":
                    mask[i] = False
                    break

        if next_parent.__class__.__name__ == "Exp":
            for i in range(self.library.get_size()):
                if self.library.get_node(i).__class__.__name__ == "Log":
                    mask[i] = False
                    break

        # if the parent is unary
        if next_parent.num_children == 1:
            #do not allow a constant
            for i in range(self.library.get_size()):
                if self.library.get_node(i).__class__.__name__ == "Const":
                    mask[i] = False
                    break

        # c<-binary->(?) If binary operator has one constant child then cannot have another
        if next_parent.num_children == 2 and next_parent.remaining_children() == 1 and next_parent.children[0].__class__.__name__ == "Const":
            for i in range(self.library.get_size()):
                if self.library.get_node(i).__class__.__name__ == "Const":
                    mask[i] = False
                    break

        if next_parent.has_trig_ancestor():
            for i in range(self.library.get_size()):
                if self.library.get_node(i).trig_ancestor == True:
                    mask[i] = False

        return mask