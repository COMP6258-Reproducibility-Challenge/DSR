import torch
from torch import nn
from torch.distributions.categorical import Categorical

class Regressor(nn.Module):
    """
    Model to carry out sybolic regression
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
        pass

    def forward(self, x, hidden_state):
        """
        Does a forward pass through the LSTM model innit
        Params
            x: int | tensor
                The indices of the parent and sibling nodes
            hidden_state: torch.tensor
                The hidden state the model outputted the previous timestep
        
        Returns 
            Tensor representing a probability distribution over actions
        """
        pass

    def sample_action(self, x, hidden_state, mask):
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
            The generated action, the log probability of choosing that action, and the entropy of the 
            generated categorical distribution 
        """
        action_dist = self.forward(x, hidden_state)
        # Use mask to zero out probabilities here
        #
        dist = Categorical(action_dist)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # Not sure if we should be calculating entropy before or after applying the mask...
        # Or if that even makes a difference
        entropy = dist.entropy()
        return action, log_prob, entropy
