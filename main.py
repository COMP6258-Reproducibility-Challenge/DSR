# import matplotlib.pyplot as plt

from models import Regressor
from Learner import Learner
from environment.SREnv import SymbolicRegressionEnv
from environment.NodeLibrary import Library
from environment.Dataset import Dataset
from environment.Expr import Expr
from environment.ExprTree import ExprTree
from environment.nodes.Node import *

embedding_size = 32
hidden_size = 32
model = Regressor(32, 32, 17)

nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X, Y, Const]
library = Library(nodes_list)

expression = ExprTree(library)
expression.add_node(0)
expression.add_node(8)
expression.add_node(9)
target_expr = Expr(library, expression.node_list)
dataset = Dataset(target_expr, numpoints=20)

env = SymbolicRegressionEnv(library, dataset, hidden_size)

learner = Learner(env, model, epochs=20)

losses, rewards, best_expr, max_reward = learner.train()