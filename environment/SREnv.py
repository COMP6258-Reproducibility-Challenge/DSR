import gymnasium as gym
import numpy as np
import torch

from gymnasium import spaces

from .Expr import Expr
from .ExprTree import ExprTree
from .NodeLibrary import Library
from .Dataset import Dataset

"""
Defines environment for symbolic regression. This is the main entry-point to the whole "environment" package, and the 
main thing our reinforcement learning implementation will interface with.
"""

class SymbolicRegressionEnv(gym.Env):
    def __init__(self, library: Library, dataset: Dataset, hidden_shape=64, device=torch.device("cpu")) -> None:
        self.expr_tree = ExprTree(library)
        self.expr_tree.node_list = []
        self.dataset = dataset
        self.library = library
        self.action = None
        self.obs = {'parent' : None, 'sibling' : None, 'hidden_state' : None}
        # At each step the model outputs a node (integer) and the next hidden state (vector)
        self.action_space = spaces.Dict({"node": spaces.Discrete(library.get_size()),
                                         "hidden_state": spaces.Box(low=-torch.inf, high=torch.inf, shape=(hidden_shape,))})

        # At each step the model receives the current node's parent, sibling, and last hidden state
        self.observation_space = spaces.Dict({"parent": spaces.Discrete(library.get_size()),
                                              "sibling": spaces.Discrete(library.get_size()),
                                              "hidden_state": spaces.Box(low=-torch.inf, high=torch.inf, shape=(hidden_shape,))})
        self.hidden_shape = hidden_shape
        self.device=device

    def _get_obs(self):
        """
        Returns the current state of the environment. This includes the parent and sibling of the current node, 
        and the hidden state produced by the RNN in the previous step
        """
        return self.obs

    def _get_info(self):
        """
        Returns the mask for constraining the search space
        
        Returns:
            mask: dict{mask : list[bool]}
                A dictionary with key "mask" and value of a mask. A mask is a list of booleans of the same length as the library
                The booleans represent whether the corresponding node from the library is valid in the next step
        """
        if not self._is_terminated():
            return {"mask": self.expr_tree.valid_nodes_mask()}
        else:
            return {"mask": [True for _ in range(self.library.get_size())]}

    def _is_terminated(self) -> bool:
        """
        Returns True if the model has finished outputting and the equation tree is complete; otherwise false
        """
        if len(self.expr_tree.node_list) > 0 and len(self.expr_tree.stack) == 0:
            return True
        return False

    def _calculate_reward(self):
        """
        Calculate reward based on current (assume finished) equation
        """
        
        expr = Expr(self.library, self.expr_tree.node_list)
        expr.optimise_consts(self.dataset)
        return self.dataset.reward(expr)

    def reset(self):
        """
        Resets the environment (gets called at the end of every episode)
        """


        self.expr_tree = ExprTree(self.library)
        self.expr_tree.node_list = []
        self.action = None
        self.obs = {'parent' : -1, 'sibling' : -1, 'hidden_state' : (torch.zeros(self.hidden_shape, device=self.device),
                                                                     torch.zeros(self.hidden_shape,device=self.device))}

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """
        Computes the environment after the model has taken an action
        Params:
            action: dictionary including next node to add and hidden state produced 
        """

        #update the observation
        self.action = action
        self.update_obs()
        #add node to the expression
        self.expr_tree.add_node(action['node'])

        terminated = self._is_terminated()
        reward = self._calculate_reward() if terminated else 0
        observation = self._get_obs()
        info = self._get_info()
        return (observation, reward, terminated, False, info)

    def update_obs(self):
        """
        Updates the observation based on the action that has been taken by the model
        """
        parent = self.library.get_node_int(self.expr_tree.get_parent_node())
        sibling = self.library.get_node_int(self.expr_tree.get_sibling_node())

        hidden_state = self.action['hidden_state']
        self.obs = {'parent' : parent, 'sibling' : sibling, 'hidden_state' : hidden_state}