from utils import * 
# import dependencies
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
import numpy as np
import gym
import itertools
from collections import deque
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

def loadAlgo(algo, algoArgs=[]):
    if algo=='BASE':
        return BASE(*algoArgs)
    if algo=='VPG':
        return VPG(*algoArgs)
    if algo=='DQN':
        return DQN(*algoArgs)
    if algo=='PPO1':
        return PPO1(*algoArgs)
    if algo=='PPO2':
        return PPO2(*algoArgs)
    if algo=='SAC':
        return SAC(*algoArgs) 
    if algo=='DDPG':
        return DDPG(*algoArgs)  

        
class BASE:
    def __init__(self, numStates, numActions=2):
        super(BASE, self).__init__()
        self.numActions = numActions
        self.numStates = numStates
        self.done = False

    def act(self, state):
        action = np.random.uniform(-1,1, 1)
        return action.item()
        
    def observe(self, state, action, reward, newState, done):
        pass 
    
    def train(self):
        pass

class VPG(nn.Module):
    def __init__(self, numStates, episodeLength = 10, numActions=10):
        super(VPG, self).__init__()
        self.rewards = []
        self.log_probs = []
        self.numActions = numActions
        self.numStates = numStates
        self.gamma = 0.99
        self.fc1 = nn.Linear(numStates, 256)
        self.fc2 = nn.Linear(256, numActions)
        self.episodeLength = episodeLength
        self.optimizer = optim.Adam(self.parameters())
        self.done = False
        self.eps = np.finfo(np.float32).eps.item()
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
        
    def act(self, state):
        state = torch.tensor(state).unsqueeze(0).float()
        probs = self.forward(state)
        multinomial = Categorical(probs)
        action = multinomial.sample()
        self.log_probs.append(multinomial.log_prob(action))
        return action.item()
        
    def observe(self, state, action, reward, newState, done):
        self.rewards.append(reward)
        self.done = done
    
    def train(self, state, action, reward, newState, done):
        if done and len(self.rewards)>=self.episodeLength:
            R = 0
            rewards = []
            for r in self.rewards[::-1]:
                R = r + self.gamma * R
                rewards.insert(0, R)
            rewards = torch.tensor(rewards)
            #rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
            policy_loss = []  
            for log_prob, reward in zip(self.log_probs, rewards):
                policy_loss.append(-log_prob * reward)
            self.optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
            del self.rewards[:]
            del self.log_probs[:]

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class PPO1(nn.Module):
    def __init__(self, num_states, learning_rate, gamma, lmbda, eps_clip, K_epoch, rollout_len, buffer_size, minibatch_size, verbose):
        super(PPO1, self).__init__()
        self.data = []
        self.num_states = num_states
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.rollout_len = rollout_len
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.verbose = verbose
        self.fc1   = nn.Linear(num_states,128)
        self.fc_mu = nn.Linear(128,1)
        self.fc_std  = nn.Linear(128,1)
        self.fc_v = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(self.buffer_size):
            for i in range(self.minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])
                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                          torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                          torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            data.append(mini_batch)
        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + self.gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))
        return data_with_adv
        
    def train(self):
        if len(self.data) == self.minibatch_size * self.buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)
            for i in range(self.K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch
                    mu, std = self.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target)
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1
        