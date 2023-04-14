from environment.nodes import Node


def BasicTest():
    y = Node.Add(None)
    a = Node.Const(y, 5)
    b = Node.Y(y)
    y.add_child(a)
    y.add_child(b)
    print(y.compute())
