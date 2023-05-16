from environment.NodeLibrary import Library
from environment.Dataset import Dataset
from environment.Expr import Expr
from environment.ExprTree import ExprTree
from environment.nodes.Node import *


# x^3 + x^2 + x
def Nguyen_1():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(0)
    expression.add_node(0)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=-1, ub=1)

    return expression, library, dataset


# x^4 + x^3 + x^2 + x
def Nguyen_2():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(0)
    expression.add_node(0)
    expression.add_node(0)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=-1, ub=1)

    return expression, library, dataset


# x^5 + x^4 + x^3 + x^2 + x
def Nguyen_3():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(0)
    expression.add_node(0)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(0)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(0)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=-1, ub=1)

    return expression, library, dataset


# x^6 + x^5 + x^4 + x^3 + x^2 + x
def Nguyen_4():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(0)
    expression.add_node(0)
    expression.add_node(0)
    expression.add_node(0)
    expression.add_node(0)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=-1, ub=1)

    return expression, library, dataset


# sin(x^2) * cos(x) - 1
def Nguyen_5():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(1)
    expression.add_node(2)
    expression.add_node(4)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(5)
    expression.add_node(8)
    expression.add_node(3)
    expression.add_node(8)
    expression.add_node(8)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=-1, ub=1)

    return expression, library, dataset


# sin(x) + sin(x + x^2)
def Nguyen_6():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(0)
    expression.add_node(4)
    expression.add_node(8)
    expression.add_node(4)
    expression.add_node(0)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=-1, ub=1)

    return expression, library, dataset


# log(x+1) + log(x^2 + 1)
def Nguyen_7():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(0)
    expression.add_node(6)
    expression.add_node(0)
    expression.add_node(8)
    expression.add_node(3)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(6)
    expression.add_node(0)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(3)
    expression.add_node(8)
    expression.add_node(8)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=0, ub=2)

    return expression, library, dataset


# sqrt(x)
def Nguyen_8():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(7)
    expression.add_node(2)
    expression.add_node(3)
    expression.add_node(8)
    expression.add_node(0)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(6)
    expression.add_node(8)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=0, ub=4)

    return expression, library, dataset


# sin(x) + sin(y^2)
def Nguyen_9():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X, Y]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(0)
    expression.add_node(4)
    expression.add_node(8)
    expression.add_node(4)
    expression.add_node(2)
    expression.add_node(9)
    expression.add_node(9)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=0, ub=1)

    return expression, library, dataset


# 2sin(x)cos(y)
def Nguyen_10():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X, Y]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(2)
    expression.add_node(3)
    expression.add_node(0)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(4)
    expression.add_node(8)
    expression.add_node(5)
    expression.add_node(9)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=0, ub=1)

    return expression, library, dataset


# x^y
def Nguyen_11():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X, Y]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(7)
    expression.add_node(2)
    expression.add_node(9)
    expression.add_node(6)
    expression.add_node(8)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=0, ub=1)

    return expression, library, dataset


# x^4 - x^3 + 0.5y^2 - y
def Nguyen_12():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X, Y]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(0)
    expression.add_node(1)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(1)
    expression.add_node(2)
    expression.add_node(3)
    expression.add_node(8)
    expression.add_node(0)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(9)
    expression.add_node(9)
    expression.add_node(9)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=0, ub=1)

    return expression, library, dataset


# 3.39x^3 + 2.12x^2 + 1.78x
def Nguyen_1_const():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X, Const]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(0)
    expression.add_node(2)
    expression.add_node(9)
    expression.node_list[-1].set_value(3.39)
    expression.add_node(2)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(0)
    expression.add_node(2)
    expression.add_node(9)
    expression.node_list[-1].set_value(2.12)
    expression.add_node(2)
    expression.add_node(8)
    expression.add_node(8)
    expression.add_node(2)
    expression.add_node(9)
    expression.node_list[-1].set_value(1.78)
    expression.add_node(8)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=-1, ub=1)

    return expression, library, dataset


# sin(1.5x) cos(0.5y)
def Nguyen_10_const():
    nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X, Y, Const]
    library = Library(nodes_list)
    expression = ExprTree(library)

    expression.add_node(2)
    expression.add_node(4)
    expression.add_node(2)
    expression.add_node(10)
    expression.node_list[-1].set_value(1.5)
    expression.add_node(8)
    expression.add_node(5)
    expression.add_node(2)
    expression.add_node(10)
    expression.node_list[-1].set_value(0.5)
    expression.add_node(9)

    target_expr = Expr(library, expression.node_list)
    print(f"Target: {target_expr}")
    dataset = Dataset(target_expr, numpoints=20, lb=0, ub=1)

    return expression, library, dataset
