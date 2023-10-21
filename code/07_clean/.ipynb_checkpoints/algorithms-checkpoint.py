import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def loadAlgo(algo, numStates, numActions, algoArgs=[]):
    if algo=='REINFORCE':
        return REINFORCE(numStates, numActions, *algoArgs)
    if algo=='REINFORCE':
        return REINFORCE(numStates, numActions, *algoArgs)
    if algo=='REINFORCE':
        return REINFORCE(numStates, numActions, *algoArgs)

class REINFORCE(nn.Module):
    def __init__(self, numStates, numActions=10, learningRate=0.0002, gamma=0.98, hiddenUnits=32):
        super(REINFORCE, self).__init__()
        self.memory = []
        self.memory_probs = []
        self.learningRate = learningRate
        self.gamma = gamma
        self.numStates = numStates
        self.numActions = numActions
        self.fc1 = nn.Linear(self.numStates, hiddenUnits)
        self.fc2 = nn.Linear(hiddenUnits, self.numActions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learningRate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
 
    def act(self, state):
        prob = self.forward(torch.from_numpy(state).float())
        m = Categorical(prob)
        action = m.sample()
        self.memory_probs.append(prob[action])
        self.action = action/(self.numActions-1)
        return self.action.item()

    def train(self):
        episodeReturn = 0
        self.optimizer.zero_grad()
        for idx, (state, action, reward, newState, done) in enumerate(self.memory[::-1]):
            prob = self.memory_probs[::-1][idx]
            episodeReturn = reward + self.gamma * episodeReturn
            loss = -torch.log(prob) * episodeReturn
            loss.backward()
        self.optimizer.step()
        self.memory = []
        self.memory_probs = []

