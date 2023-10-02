import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Algorithm:
    def __init__(self, numStates, numActions):



class REINFORCE(nn.Module):
    def __init__(self, numStates, numActions):
        super(REINFORCE, self).__init__()
        self.learningRate = 0.0002
        self.gamma = 0.98
        self.data = []
        self.fc1 = nn.Linear(numStates, 128)
        self.fc2 = nn.Linear(128, numActions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learningRate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
        
    def train(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob) * R
            loss.backward(retain_graph=True)
        self.optimizer.step()
        self.data = []