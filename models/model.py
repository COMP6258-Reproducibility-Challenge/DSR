import torch
from torch import nn
from torch.distributions.categorical import Categorical
import numpy as np


class Regressor(nn.Module):
    """
    Model to carry out symbolic regression
    """

    def __init__(self, embedding_size, hidden_size, output_size):
        """
        Defines the LSTM and embedding model
        Params
            embedding_size: int
                The size of the embedding vector of the nodes
            hidden_size: int
                The size of the hidden activations
            output_size: int
                The number of possible nodes to output

        """
        super().__init__()

        self.output_size = output_size
        self.embedding = torch.empty(output_size, embedding_size)
        nn.init.kaiming_normal_(self.embedding)
        self.lstm = nn.LSTM(embedding_size, hidden_size)

    def forward(self, x, hidden_state):
        """
        Does a forward pass through the LSTM model innit
        Params
            x: int | tensor
                The indices of the parent and sibling nodes
            hidden_state: torch.tensor
                The hidden state the model outputted the previous timestep
        
        Returns
            Tuple of probability distribution and hidden state
        """

        LSTM_output, last_hidden_state = self.lstm(self.embedding[x], hidden_state)
        softmax = nn.Softmax(dim=self.output_size)
        prob_dist = softmax(LSTM_output)

        return prob_dist, last_hidden_state

    def sample_action(self, observation, mask):
        """
        Samples an action given an observation
        Params
            x: torch.tensor
                The concatenated indices of the sibling and parent node of current node
            hidden_state: torch.tensor
                The hidden state the model outputted the previous timestep
            mask: torch.tensor
                Denotes valid actions
        
        Returns
            Tuple of the generated action, the hidden state, the log probability of choosing that action, and the entropy of the
            generated categorical distribution
        """
        x = torch.concat((observation["parent"], observation["sibling"]))
        hidden_state = observation["hidden_state"]
        action_dist, next_hidden_state = self.forward(x, hidden_state)
        # Use mask to zero out probabilities here
        # those are of the invalid nodes
        action_dist *= torch.as_tensor(mask).reshape(action_dist.shape)
        action_dist /= action_dist.sum()

        dist = Categorical(action_dist)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        entropy = dist.entropy()
        return action, next_hidden_state, log_prob, entropy
