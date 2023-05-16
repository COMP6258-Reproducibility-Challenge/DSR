from abc import ABC, abstractmethod
import torch
from torch.distributions.categorical import Categorical

from environment.ExprTree import ExprTree

class Loss(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def calculate(self, log_probs, entropies, rewards, exprs):
        pass


class RSPGLoss(Loss):
    """
    Risk Seeking Policy Gradient Loss. Instead of weighting the function with ewma it takes
    the best working functions of a particular quantile and modifies the probability distribution 
    which has all of the log probabilties guaranted to be greater than zero
    """
    def __init__(self, risk_factor=0.05, entropy_coef=0.005):
        """
        Params
            risk_factor: float
                quantile of the functions that are kept.
            entropy_coef: float
                entropy coeficient of the REINFORCE algoirthmn
        """
        self.risk_factor = risk_factor
        self.entropy_coef = entropy_coef

    def calculate(self, log_probs, entropies, rewards, exprs):
        """
        Calculates the loss
        
        Params:
            log_probs: tensor(float)
                log probability of each fucntion generated in the epoch
            entropies: tensor(float)
                entropy of each fucntion generated in the epoch
            rewards: tensor(float)
                reward of each fucntion generated in the epoch
        Returns:
            A dictionary containing:
            log_prob_term: tensor(float)
                mean log probability of each distibution produced at each timestep
            entropy_term: tensor(float)
                mean entropies of each distibution produced at each timestep
            loss: tensor(float)
                composed of the negative(log_prob_term + entropy_term)
            max_reward: tensor(float)
                max reward of the epoch
            max_reward_i: int
                index of function with max reward
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
    
class VPGLoss(Loss):
    """
    Vanila Policy Gradient Loss function as defined by the REINFORCE algorithmn
    """
    
    def __init__(self, beta=0.25, entropy_coef=0.005):
        """
        Params
            beta: float
                beta of the REINFORCE algorithmn
            entropy_coef: float
                entropy coeficient of the REINFORCE algoirthmn
        """
        self.beta = beta
        self.entropy_coef = entropy_coef
        self.ewma = 0

    def calculate(self, log_probs, entropies, rewards, exprs):
        """"
        Calculates the loss
        
        Params:
            log_probs: tensor(float)
                log probability of each fucntion generated in the epoch
            entropies: tensor(float)
                entropy of each fucntion generated in the epoch
            rewards: tensor(float)
                reward of each fucntion generated in the epoch
        Returns:
            A dictionary containing:
            log_prob_term: tensor(float)
                mean log probability of each distibution produced at each timestep
            entropy_term: tensor(float)
                mean entropies of each distibution produced at each timestep
            loss: tensor(float)
                composed of the negative(log_prob_term + entropy_term)
            max_reward: tensor(float)
                max reward of the epoch
            max_reward_i: int
                index of function with max reward
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

class PQTLoss(Loss):
    """
    Priority Queue Training is maintaining a queue of the top k highest rewards with the objective
    being the average log-likelihood of the samples in the priority queue
    """
    def __init__(self, model, library, k=10, entropy_coef=0.005, device=torch.device("cpu")):
        """
        Params
            model: Model
                model for symbolic regression
            library: Library
                library with the expression list, dataset and target expression
            k: int
                length of priority queue
            entropy_coef: float
                entropy coeficient of the REINFORCE algoirthmn
            device: torch.device
                device that is running the project
        """
        # Reward, Exprs
        self.pq = [torch.full((k,), -1, device=device), []]
        self.device = device
        self.entropy_coef = entropy_coef
        self.k = k
        self.model = model
        self.library = library

    def calculate(self, log_probs, entropies, rewards, exprs):
        """"
        Calculates the loss
        
        Params:
            log_probs: tensor(float)
                log probability of each fucntion generated in the epoch
            entropies: tensor(float)
                entropy of each fucntion generated in the epoch
            rewards: tensor(float)
                reward of each fucntion generated in the epoch
        Returns:
            A dictionary containing:
            log_prob_term: tensor(float)
                mean log probability of each distibution produced at each timestep
            entropy_term: tensor(float)
                mean entropies of each distibution produced at each timestep
            loss: tensor(float)
                composed of the negative(log_prob_term + entropy_term)
            max_reward: tensor(float)
                max reward of the epoch
            max_reward_i: int
                index of function with max reward
        """
        pq_log_probs, pq_entropies = self.calc_probs(self.pq[1])

        all_rewards = torch.cat([rewards, self.pq[0]])        
        all_log_probs = torch.cat([log_probs, pq_log_probs])
        all_entropies = torch.cat([entropies, pq_entropies])
        all_exprs = exprs + self.pq[1]

        self.pq[0], best_i = torch.topk(all_rewards, self.k)
        selected_log_probs = all_log_probs[best_i]
        self.pq[1] = [all_exprs[i] for i in best_i]
        
        not_selected = torch.ones_like(all_log_probs, dtype=torch.int)
        not_selected[best_i] = 0
        all_log_probs[not_selected].detach()

        g1 = selected_log_probs.mean()
        g2 = self.entropy_coef * entropies.mean()

        max_reward = torch.max(rewards)
        max_reward_i = torch.argmax(rewards)   

        # The loss term is negative here because pytorch optimisers by default perform gradient descent, but
        # we want to do gradient ascent on g1+g2
        return {"log_prob_term": g1, "entropy_term": g2, "loss": -(g1 + g2),
                "max_reward": max_reward, "max_reward_i": max_reward_i}

    def calc_probs(self, exprs):
        """
        Calculates the log probabilty and entropies of each of the expressions in the queue
        
        Params:
            expr: list(expr)
                List of all the expressions
        
        Returns:
            A tuple containing:
            log_probs: tensor(float)
                log probabilties of each expression 
            entropies: tensor(float)
                entropies of each expression 
        """
        log_probs = torch.empty((self.k,), device=self.device)
        entropies = torch.empty((self.k,), device=self.device)
        # Each expr is an expression tree
        for i, expr in enumerate(exprs):
            log_prob = 0
            entropy = 0

            hidden_state = None
            parent = -1
            sibling = -1
            prob_dist, hidden_state = self.model.forward(parent, sibling, hidden_state)

            expr_tree = ExprTree(self.library)
            expr_tree.node_list = []
            mask = torch.tensor(expr_tree.valid_nodes_mask(), device=self.device)
            for node in expr.node_list:
                j = self.library.get_node_int(node)

                entropy += torch.sum(-prob_dist * torch.log(prob_dist), dim=-1)

                masked_dist = prob_dist * mask
                normalised_masked_dist = masked_dist / torch.sum(masked_dist, dim=-1)#.unsqueeze(1)
                log_prob += torch.log(normalised_masked_dist[j])
                
                expr_tree.add_node(j)
                if len(expr_tree.stack) == 0:
                    break
                mask = torch.tensor(expr_tree.valid_nodes_mask(), device=self.device)
                parent = self.library.get_node_int(expr_tree.get_parent_node())
                sibling = self.library.get_node_int(expr_tree.get_sibling_node())
                prob_dist, hidden_state = self.model.forward(parent, sibling, hidden_state)
                
            log_probs[i] = log_prob
            entropies[i] = entropy
        return log_probs, entropies