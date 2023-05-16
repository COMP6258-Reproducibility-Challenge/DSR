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
        batched_infos = {"mask": []}
        for info in infos:
            batched_infos["mask"].append(info["mask"])
        
        batched_infos["mask"] = torch.tensor(batched_infos["mask"], device=self.device)
        
        return batched_infos

    def unbatch_actions(self, action_dict):
        new_action_dict = []
        for action, h, c in zip(action_dict["node"], action_dict["hidden_state"][0], action_dict["hidden_state"][1]):
            new_single_dict = {"node": action, "hidden_state": (h,c)}
            new_action_dict.append(new_single_dict)

        return new_action_dict

    def get_exprs(self):
        return [e.expr_tree for e in self.envs]
    
    def filter_obs(self, obs, done):
        new_obs = {}
        new_obs["parent"] = obs["parent"][~done]
        new_obs["sibling"] = obs["sibling"][~done]
        new_obs["hidden_state"] = (
            obs["hidden_state"][0][~done],
            obs["hidden_state"][1][~done]
        )
        return new_obs
