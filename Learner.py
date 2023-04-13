import torch
from environment import SREnv, NodeLibrary, Dataset, Expr

class Learner():

    def __init__(self, env: SREnv, model, library: NodeLibrary, dataset: Dataset, risk_factor=0.05, entropy_coef=0.005, 
                 epochs=10, batch_size=1000):
        self.env = env
        self.model = model
        self.library = library
        self.dataset = dataset

        self.risk_factor = risk_factor
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size

    def loss(self, log_probs, entropies, rewards):
        """
        Params
            log_probs: torch.Tensor
                Log probabilities of expressions being sampled. Shape (N,)
            entropies: torch.Tensor
                Entropy of generated categorical distributions for each expression. Shape (N,)
            rewards: torch.Tensor
                Rewards of each generated expression. Shape (N,)
            risk_factor: float
                The proportion of expressions that are kept 
            entropy_coefficient: float
                The coefficient for the entropy loss term
        Returns
            A dictionary holding different loss terms and other auxilliary info
        """
        r_eps = torch.quantile(rewards, 1-self.risk_factor, interpolation='linear')
        keep = rewards > r_eps
        rewards_keep = rewards[keep]
        log_probs_keep = log_probs[keep]
        entropies_keep = entropies[keep]

        g1 = ((rewards_keep - r_eps) * log_probs_keep).mean()
        g2 = self.entropy_coef * entropies_keep.mean()

        max_reward = torch.max(rewards)
        max_reward_i = torch.argmax(rewards)

        return {"log_prob_term": g1, "entropy_term": g2, "loss": g1 + g2,
                "max_reward": max_reward, "max_reward_i": max_reward_i}
    
    def update(self):
        log_probs, entropies, rewards, exprs = self.get_batch()
        loss_dict = self.loss(log_probs, entropies, rewards)
        # calculate loss over batch and back_prop
        # generate best expression so far
        pass
    
    def train(self):
        # for each epoch
        #   call update
        #   get best performing expr per generation
        pass

    def get_batch(self):
        rewards = torch.empty((self.batch_size,))
        probs = torch.empty((self.batch_size,))
        entropies = torch.empty((self.batch_size,))
        for i in range(self.batch_size):
            # generate expression
            # for each token generated  
            #   running_entropy += entropy
            #   log_prob += log(p)
            # get reward
            pass
        # return rewards, entropies, probs, exprs
    
    def generate_best_expr(self, node_index_list):
        # Return an Expr of the best generated
        pass
        

    