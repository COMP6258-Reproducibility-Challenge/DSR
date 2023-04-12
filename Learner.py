import torch

class Learner():

    def __init__(self, env, model, library, dataset):
        self.env = env
        self.model = model
        self.library = library
        self.dataset = dataset

    def loss(self, actions, log_probs, entropies, rewards):
        pass
    
    def update(self):
        pass
    
    def train(self, epochs):
        pass

    