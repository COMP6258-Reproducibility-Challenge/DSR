from nodes import Node
from NodeLibrary import Library
from Expr import Expr


def BasicTest():
    nodes = [Node.Add(), Node.Sub(), Node.Mult(), Node.Div(), Node.Sin(
    ), Node.Cos(), Node.Log(), Node.Exp(), Node.X(), Node.Y(), Node.Const()]
    lib = Library(nodes)
    x = Expr(lib)
    for y in [3, 4, 2, 10, 8, 6, 9]:
        x.add_node(y)
    print(repr(x))


def Constraint():
    nodes = [Node.Add(), Node.Sub(), Node.Mult(), Node.Div(), Node.Sin(
    ), Node.Cos(), Node.Log(), Node.Exp(), Node.X(), Node.Y(), Node.Const()]
    lib = Library(nodes)
    x = Expr(lib)
    for y in [0, 4, 2, 10, 9, 8]:
        print(x.valid_nodes_mask())
        x.add_node(y)
    print(repr(x))


Constraint()
