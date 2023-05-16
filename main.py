from Benchmarks import *
from Learner import Learner
from Losses import *
from environment.BatchEnv import BatchEnv
from models import Regressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The Benchmark class instantiates all relevant Nguyen expression
# which return the expression, its library, and the corresponding dataset
expression, library, dataset = Nguyen_1()

embedding_size = 32
hidden_size = 32
model = Regressor(embedding_size, hidden_size, library.get_size(), device=device)

env = BatchEnv(library, dataset, hidden_size, batch_size=1000, device=device)

# defining the loss - PQTLoss(model, library, device), VPGLoss(), and RSPGLoss())
loss = RSPGLoss()

learner = Learner(env, model, loss, epochs=2000, batch_size=1000, device=device)
losses, rewards, best_expr, max_reward = learner.train()
