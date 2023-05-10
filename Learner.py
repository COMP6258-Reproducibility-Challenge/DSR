import torch
from torch import optim

from environment import SREnv, NodeLibrary, Dataset, Expr

class Learner():
    """
    Runs the main body of the algortithm, including model learning and generating the best expressions
    """
    def __init__(self, env, model, risk_factor=0.05, entropy_coef=0.005, epochs=2000, batch_size=1000,
                 lr=0.0005, beta=0.25, k=10, method="rspg", device=torch.device("cpu")):
        self.env = env
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr=lr)

        self.risk_factor = risk_factor
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta = beta
        self.k = k
        self.method = method

        self.ewma = 0

        # Priority Queue, first is reward, second is logprobs
        self.pq = [torch.full((k,), -1, device=device), torch.zeros((k,), device=device)]

        self.device = device

    def loss_rspg(self, log_probs, entropies, rewards):
        """
        Params
            log_probs: torch.Tensor
                Log probabilities of expressions being sampled. Shape (N,)
            entropies: torch.Tensor
                Entropy of generated categorical distributions for each expression. Shape (N,)
            rewards: torch.Tensor
                Rewards of each generated expression. Shape (N,)
        Returns
            A dictionary holding different loss terms and other auxilliary info
        """
        # Filter the low performing rewards, keeping only the top quantile of expressions
        r_eps = torch.quantile(rewards, 1-self.risk_factor, interpolation='linear')
        keep = rewards >= r_eps
        rewards_keep = rewards[keep]
        log_probs_keep = log_probs[keep]
        entropies_keep = entropies[keep]

        g1 = ((rewards_keep - r_eps) * log_probs_keep).mean()
        g2 = self.entropy_coef * entropies_keep.mean()

        max_reward = torch.max(rewards)
        max_reward_i = torch.argmax(rewards)    

        # The loss term is negative here because pytorch optimisers by default perform gradient descent, but
        # we want to do gradient ascent on g1+g2
        return {"log_prob_term": g1, "entropy_term": g2, "loss": -(g1 + g2),
                "max_reward": max_reward, "max_reward_i": max_reward_i}

    def loss_vpg(self, log_probs, entropies, rewards):
        """
        Params
            log_probs: torch.Tensor
                Log probabilities of expressions being sampled. Shape (N,)
            entropies: torch.Tensor
                Entropy of generated categorical distributions for each expression. Shape (N,)
            rewards: torch.Tensor
                Rewards of each generated expression. Shape (N,)
        Returns
            A dictionary holding different loss terms and other auxilliary info
        """
        # Filter the low performing rewards, keeping only the top quantile of expressions
        self.ewma = self.beta * rewards.mean() + (1 - self.beta) * self.ewma

        g1 = ((rewards - self.ewma) * log_probs).mean()
        g2 = self.entropy_coef * entropies.mean()

        max_reward = torch.max(rewards)
        max_reward_i = torch.argmax(rewards)    

        # The loss term is negative here because pytorch optimisers by default perform gradient descent, but
        # we want to do gradient ascent on g1+g2
        return {"log_prob_term": g1, "entropy_term": g2, "loss": -(g1 + g2),
                "max_reward": max_reward, "max_reward_i": max_reward_i}
    
    def loss_pqt(self, log_probs, entropies, rewards):
        """
        Params
            log_probs: torch.Tensor
                Log probabilities of expressions being sampled. Shape (N,)
            entropies: torch.Tensor
                Entropy of generated categorical distributions for each expression. Shape (N,)
            rewards: torch.Tensor
                Rewards of each generated expression. Shape (N,)
        Returns
            A dictionary holding different loss terms and other auxilliary info
        """
        # Filter the low performing rewards, keeping only the top quantile of expressions

        all_rewards = torch.cat([rewards, self.pq[0]])
        all_log_probs = torch.cat([log_probs, self.pq[1]])
        self.pq[0], best_i = torch.topk(all_rewards, self.k)
        self.pq[1] = all_log_probs[best_i]

        g1 = self.pq[0].mean()
        g2 = self.entropy_coef * entropies.mean()

        max_reward = torch.max(rewards)
        max_reward_i = torch.argmax(rewards)    

        # The loss term is negative here because pytorch optimisers by default perform gradient descent, but
        # we want to do gradient ascent on g1+g2
        return {"log_prob_term": g1, "entropy_term": g2, "loss": -(g1 + g2),
                "max_reward": max_reward, "max_reward_i": max_reward_i}
    
    def get_multi_batch(self):
        obs, info = self.env.reset()
        mask = info["mask"]
        done = torch.full((self.batch_size, ), False, device=self.device)
        probs = torch.zeros((self.batch_size,), device=self.device)
        entropies = torch.zeros((self.batch_size,), device=self.device)
        while torch.any(done == False):
            # need to filter out dones here
            # pobs = self.env.filter_obs(obs, done)
            action, hidden, log_prob, entropy = self.model.sample_action(obs, mask)
            probs[~done] += log_prob
            entropies[~done] += entropy

            action_dict = {"node": action, "hidden_state": hidden}
            action_dict = self.env.unbatch_actions(action_dict)
            obs, reward, done, info = self.env.step(action_dict)
            mask = info["mask"]
        reward = torch.nan_to_num(reward, 0)
        return reward, entropies, probs, self.env.get_exprs()
    
    def update(self):
        """
        Generates a single batch and performs gradient ascent on the loss term.

        Returns:
            The best expression of the batch, and the generated loss dictionary of the batch
        """
        self.optim.zero_grad()
        rewards, entropies, log_probs, exprs = self.get_multi_batch()

        if self.method == "rspg":
            loss_dict = self.loss_rspg(log_probs, entropies, rewards)
        elif self.method == "vpg":
            loss_dict = self.loss_vpg(log_probs, entropies, rewards)
        elif self.method == "pqt":
            loss_dict = self.loss_pqt(log_probs, entropies, rewards)

        loss = loss_dict["loss"]
        loss.backward()
        self.optim.step()
        best_expr = exprs[loss_dict["max_reward_i"]].build_expression()
        return best_expr, loss_dict
    
    def train(self):
        """
        The main training loop. This will keep track of the rewards and losses, as well as the maximum reward 
        generated and its associated expression. These are all then returned for graphing/analysis.
        """
        max_reward = -float("inf")
        best_expr = None
        rewards = []
        losses = []
        for epoch in range(self.epochs):
            expr, loss_dict = self.update()
            reward = loss_dict["max_reward"]
            loss = loss_dict["loss"].detach()
            rewards.append(reward)
            losses.append(loss)
            if reward > max_reward:
                max_reward = reward
                best_expr = expr
            
            # if (epoch+1) % 10 == 0:
            #     # I am hoping having best_expr inside the f-string automatically calls __repr__, not sure though lmao. 
            #     # Easy to fix if not.
            print(f"Epoch: {epoch}/{self.epochs} - Expr: {best_expr} - Reward: {reward} - Loss: {loss}")
            if reward == 1.0:
                break
        
        return losses, rewards, best_expr, max_reward
    