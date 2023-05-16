import gymnasium as gym
import numpy as np
import torch

from .SREnv import SymbolicRegressionEnv

class BatchEnv():
    """
    Defines a batch of environments to be used for batch learning
    """
    def __init__(self, library, dataset, hidden_size, batch_size=1000, device=torch.device("cpu")):
        self.envs = [SymbolicRegressionEnv(library, dataset, hidden_size, device=device) for _ in range(batch_size)]
        self.dones = torch.full((batch_size,), False, device=device)
        self.rewards = torch.empty((batch_size,), device=device)
        self.batch_size = batch_size
        self.device = device

    def reset(self):
        """
        Resets the all the environments of the batch (gets called at the end of every episode)
        """
        self.dones = torch.full((self.batch_size,), False, device=self.device)
        self.rewards = torch.empty((self.batch_size, ), device=self.device)
        observations = []
        infos = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return self.get_batch_obs(observations), self.get_batch_info(infos)
    
    def step(self, actions):
        """
        Computes the environment after the model has taken actions
        Params:
            actions: list of dictionaries including next node to add and hidden state produced 
        """
        observations = []
        infos = []
        indices = [i for i,done in enumerate(self.dones) if not done]
        envs = [env for env,done in zip(self.envs, self.dones) if not done]

        #observation, reward, terminated, False, info
        for i, action, env in (zip(indices, actions, envs)):
            if not self.dones[i]:
                obs, reward, done, _, info = env.step(action)
                self.rewards[i] = reward
                self.dones[i] = done
                if not done:
                    infos.append(info)
                    observations.append(obs)
        
        return self.get_batch_obs(observations), self.rewards, self.dones, self.get_batch_info(infos)

    def get_batch_obs(self, observations):
        """
        Returns the current states of the environments. This includes the parents and siblings of the current nodes, 
        and the hidden states produced by the RNN in the previous step
        """
        batched_obs = {"parent": [], "sibling": [], "hidden_state": []}
        if len(observations) == 0:
            return batched_obs
        for obs in observations:
            batched_obs["parent"].append(obs["parent"])
            batched_obs["sibling"].append(obs["sibling"])
            batched_obs["hidden_state"].append(obs["hidden_state"])
        batched_obs["parent"] = torch.tensor(batched_obs["parent"], device=self.device)
        batched_obs["sibling"] = torch.tensor(batched_obs["sibling"], device=self.device)
        batched_obs["hidden_state"] = (
            torch.stack([t[0] for t in batched_obs["hidden_state"]]),
            torch.stack([t[1] for t in batched_obs["hidden_state"]])
        )

        return batched_obs

    def get_batch_info(self, infos):
        """
        Returns the masks for constraining the search space
        
        Returns:
            mask: dict{mask : list[list[bool]]}
                A dictionary with key "mask" and value of a list of masks. A mask is a list of booleans of the same length as the library
                The booleans represent whether the corresponding node from the library is valid in the next step
        """
        batched_infos = {"mask": []}
        for info in infos:
            batched_infos["mask"].append(info["mask"])
        
        batched_infos["mask"] = torch.tensor(batched_infos["mask"], device=self.device)
        
        return batched_infos

    def unbatch_actions(self, action_dict):
        """
        Formats the action_dict so that it is not by batch and is by the actions instead
        
        Returns:
            new_action_dict: dict{"node": Node, "hidden_state": (tensor(float),tensor(float))}
                A dictionary with key "node" and value of the associated node with the associated "hidden_state" of a
                tuple with ("hidden state","context state") (tensor(float),tensor(float))
        """
        new_action_dict = []
        for action, h, c in zip(action_dict["node"], action_dict["hidden_state"][0], action_dict["hidden_state"][1]):
            new_single_dict = {"node": action, "hidden_state": (h,c)}
            new_action_dict.append(new_single_dict)

        return new_action_dict

    def get_exprs(self):
        """
        Gets a list of the expresion trees
        
        Returns:
            list[ExprTree]
                A list of expression trees for the whole batch
        """
        return [e.expr_tree for e in self.envs]
    
    def filter_obs(self, obs, done):
        """
        Removes all the completed observations
        
        Returns:
            new_obs: dict{parent : list[Node], sibling : list[Node], hidden_state : list[(tensor(float),tensor(float))]}
                The new observation that has filtered out completed observations.
        """
        new_obs = {}
        new_obs["parent"] = obs["parent"][~done]
        new_obs["sibling"] = obs["sibling"][~done]
        new_obs["hidden_state"] = (
            obs["hidden_state"][0][~done],
            obs["hidden_state"][1][~done]
        )
        return new_obs
