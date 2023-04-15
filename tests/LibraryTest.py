from nodes import Node
from NodeLibrary import Library

def BasicTest():
    nodes = [Node.Add(), Node.Sub(), Node.Mult(), Node.Div(), Node.Sin(), Node.Cos(), Node.Log(), Node.Exp(), Node.X(), Node.Y(), Node.Const()]
    lib = Library(nodes)
    lib.get_node(1).stringify()

BasicTest()