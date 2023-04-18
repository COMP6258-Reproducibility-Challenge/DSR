import gymnasium as gym
import numpy as np
import torch

from gymnasium import spaces

from Expr import Expr
from NodeLibrary import Library
from Dataset import Dataset

"""
Defines environment for symbolic regression. This is the main entry-point to the whole "environment" package, and the 
main thing our reinforcement learning implementation will interface with.
"""


class SymbolicRegressionEnv(gym.Env):
    def __init__(self, library: Library, dataset: Dataset, hidden_shape=64) -> None:
        self.expr = Expr(library)
        self.dataset = dataset
        self.library = library
        # At each step the model outputs a node (integer) and the next hidden state (vector)
        self.action_space = spaces.Dict({"node": spaces.Discrete(library.get_size()),
                                         "hidden_state": spaces.Box(low=-torch.inf, high=torch.inf, shape=(hidden_shape,))})

        # At each step the model receives the current node's parent, sibling, and last hidden state
        self.observation_space = spaces.Dict({"parent": spaces.Discrete(library.get_size()),
                                              "sibling": spaces.Discrete(library.get_size()),
                                              "hidden_state": spaces.Box(low=-torch.inf, high=torch.inf, shape=(hidden_shape,))})

    def _get_obs(self):
        """
        Returns the current state of the environment (should be an element of the observation space). This includes the 
        parent and sibling of the current node, and the hidden state produced by the RNN in the previous step
        """
        parent = self.library.get_node_int(self.expr.get_parent_node())
        sibling = self.library.get_node_int(self.expr.get_sibling_node())
        pass

    def _get_info(self):
        """
        Returns auxillary information dictionary about the environment/last step produced
        (i.e. current tree size maybe? idk we'll see)
        """
        pass

    def _is_terminated(self) -> bool:
        """
        Returns True if the model has finished outputting and the equation tree is complete; otherwise false
        """
        if len(self.expr.node_list) > 0 and len(self.expr.stack) == 0:
            return True
        return False

    def _calculate_reward(self):
        """
        Calculate reward based on current (assume finished) equation
        """
        return self.dataset.reward(self.expr)

    def reset(self):
        """
        Resets the environment (gets called at the end of every episode)
        """
        # Reset relevant properties here
        #

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """
        Computes the environment after the model has taken an action
        Params:
            action: dictionary including next node to add and hidden state produced
        """
        # Compute new environment after action has been applied here
        #

        terminated = self._is_terminated()
        reward = self._calculate_reward() if terminated else 0
        observation = self._get_obs()
        info = self._get_info()
        return (observation, reward, terminated, False, info)
