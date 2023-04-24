# import matplotlib.pyplot as plt

from models import Regressor
from Learner import Learner
from environment.SREnv import SymbolicRegressionEnv
from environment.NodeLibrary import Library
from environment.Dataset import Dataset
from environment.Expr import Expr
from environment.ExprTree import ExprTree
from environment.nodes.Node import *

nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X]#, Y]#, Const]
library = Library(nodes_list)

embedding_size = 64
hidden_size = 64
model = Regressor(embedding_size, hidden_size, library.get_size())

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
dataset = Dataset(target_expr, numpoints=20)

env = SymbolicRegressionEnv(library, dataset, hidden_size)

learner = Learner(env, model, epochs=2000, batch_size=1000)
# torch.autograd.set_detect_anomaly(True)
losses, rewards, best_expr, max_reward = learner.train()