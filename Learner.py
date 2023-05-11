import torch
from torch import optim

from environment import SREnv, NodeLibrary, Dataset, Expr

class Learner():
    """
    Runs the main body of the algortithm, including model learning and generating the best expressions
    """
    def __init__(self, env, model, loss, epochs=2000, batch_size=1000, lr=0.0005, device=torch.device("cpu")):
        self.env = env
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr=lr)
        self.loss = loss

        self.epochs = epochs
        self.batch_size = batch_size

        self.device = device

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

        loss_dict = self.loss.calculate(log_probs, entropies, rewards, exprs)

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
            print(f"Epoch: {epoch}/{self.epochs} - Expr: {best_expr} - Reward: {reward} - Loss: {loss}")
            if reward == 1.0:
                break
        
        return losses, rewards, best_expr, max_reward
    