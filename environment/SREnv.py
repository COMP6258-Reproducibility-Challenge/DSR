import gymnasium as gym
import numpy as np

from gymnasium import spaces

from EquationTree import EquationTree
from Library import Library
from Dataset import Dataset

"""
Defines environment for symbolic regression.
"""

class SymbolicRegressionEnv(gym.Env):
    def __init__(self, library: Library, dataset: Dataset, hidden_shape=64) -> None:
        self.equation_tree = EquationTree()
        self.dataset = dataset
        self.library = library
        # At each step the model outputs a node (integer) and the next hidden state (vector)
        self.action_space = spaces.Dict({"node": spaces.Discrete(library.get_size()), 
                                         "hidden_state": spaces.Box(low=-np.inf, high=np.inf, shape=(hidden_shape,))})
        
        # At each step the model receives the current node's parent, sibling, and last hidden state
        self.observation_space = spaces.Dict({"parent": spaces.Discrete(library.get_size()),
                                              "sibling": spaces.Discrete(library.get_size()),
                                              "hidden_state": spaces.Box(low=-np.inf, high=np.inf, shape=(hidden_shape,))})
    
    def _get_obs(self):
        """
        Returns the current state of the environment
        """
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
        return True
    
    def _calculate_reward(self):
        """
        Calculate reward based on current (assume finished) equation
        """
        return 1

    def reset(self):
        """
        Resets the environment (gets called at the end of every episode)
        """

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
        #
        terminated = self._is_terminated()
        reward = self._calculate_reward() if terminated else 0
        observation = self._get_obs()
        info = self._get_info()
        return (observation, reward, terminated, False, info)
         