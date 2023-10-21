###################################################################################################
#                                   SETUP
###################################################################################################

from utils import *
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gym
import itertools
from collections import deque
import torch
import torch.nn.functional as F
import random
from torch.nn import init
import collections

def loadAlgo(algo, algoArgs=[]):
    if algo=='VPGContinuous':
        return VPGContinuous(*algoArgs)
    if algo=='DQN':
        return DQN(*algoArgs)
###################################################################################################
#                                   TABULAR
###################################################################################################


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

###################################################################################################
#                                   VPG
###################################################################################################

            
class VPGContinuous(nn.Module):
    def __init__(self, numStates=2, episodeLength=10, numTrajectories=5, gamma=0.99,
                 std_init = 0.5, std_decay = 0.9995, std_min = 0.02, lr = 0.0003, k = -0.15, p = 0.3):
        super(VPGContinuous, self).__init__()
        self.memory = []
        self.rewards = []
        self.log_probs = []
        self.numStates = numStates
        self.gamma = gamma
        self.numTrajectories = numTrajectories
        self.fc1_mu = nn.Linear(numStates, 128)
        self.fc2_mu = nn.Linear(128, 64)
        self.fc3_mu = nn.Linear(64, 1)
        #self.fc1_std = nn.Linear(numStates, 128)
        #self.fc2_std = nn.Linear(128, 64)
        #self.fc3_std = nn.Linear(64, 1)
        #init.orthogonal_(self.fc1_mu.weight, gain=0.01)
        #init.orthogonal_(self.fc2_mu.weight, gain=0.01)
        init.orthogonal_(self.fc3_mu.weight, gain=0.01)
        #init.orthogonal_(self.fc1_std.weight, gain=0.01)
        #init.orthogonal_(self.fc2_std.weight, gain=0.01)
        #init.orthogonal_(self.fc3_std.weight, gain=0.01)
        self.episodeLength = episodeLength
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.done = False
        self.eps = np.finfo(np.float32).eps.item()
        self.policy_loss = []
        self.traj = 0
        self.cnt = 0
        self.k = k
        self.p = p
        self.std = std_init
        self.std_init = std_init
        self.std_decay = std_decay
        self.std_min = std_min

    def forward(self, x):
        x = x.float()
        #x = (x - self.state_mean) / (self.state_std + self.eps)
        mu = F.leaky_relu(self.fc1_mu(x))
        mu = F.leaky_relu(self.fc2_mu(mu))
        mu = F.tanh(self.fc3_mu(mu))
        #std = F.leaky_relu(self.fc1_std(x))
        #std = F.leaky_relu(self.fc2_std(std))
        #std = F.softplus(self.fc3_std(std)-0.5)  # Ensure std is positive
        return mu, None

    def observe(self, state, action, reward, newState, done):
        self.memory.append([state, self.action.item(), reward, newState, done])
        self.rewards.append(reward/100)
        self.done = done

    def act(self, state):
        state = torch.tensor(state).unsqueeze(0).float()
        mean, _ = self.forward(state)
        self.mean = mean.item()
        normal_dist = Normal(mean, self.std)
        self.action = normal_dist.sample()
        self.log_probs.append(normal_dist.log_prob(self.action))
        return self.action.item()
    
    def train(self):
        state, action, reward, newState, done = self.memory[-1]
        if done and len(self.rewards)>=self.episodeLength:
            R = 0
            rewards = []
            for r in self.rewards[::-1]:
                R = r + self.gamma * R
                rewards.insert(0, R)
            rewards = torch.tensor(rewards) 
            for log_prob, reward in zip(self.log_probs, rewards):
                self.policy_loss.append(-log_prob * reward)
            del self.rewards[:]
            del self.log_probs[:]
            self.traj += 1
            if self.traj == self.numTrajectories:
                self.optimizer.zero_grad()
                policy_loss = torch.cat(self.policy_loss).sum()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                self.optimizer.step()
                self.update_std()
                self.policy_loss = []
                self.traj = 0
                self.cnt += 1

    #def update_std(self, cnt):
    #    if self.std > self.std_min:
    #        temperature = self.std_init * (np.exp(self.k * cnt ** self.p))
    #        self.std = max(temperature, self.std_min)

    def update_std(self):
        if self.std > self.std_min:
            self.std *= self.std_decay



###################################################################################################
#                                   DQN
###################################################################################################

class Qnet(nn.Module):
    def __init__(self, numStates, numActions):
        super(Qnet, self).__init__()
        self.numStates = numStates
        self.numActions = numActions
        self.fc1 = nn.Linear(numStates, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, numActions)
        init.orthogonal_(self.fc1.weight, gain=0.01)
        init.orthogonal_(self.fc2.weight, gain=0.01)
        init.orthogonal_(self.fc3.weight, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        if random.random() < epsilon:
            return random.randint(0,self.numActions-1)
        else: 
            return out.argmax().item()

class PrioritizedBuffer():
    def __init__(self, buffer_limit, numStates, eta=1.0, chi=1e-5):
        self.buffer_limit = buffer_limit
        self.numStates = numStates
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.priorities = np.zeros(buffer_limit, dtype=np.float32)
        self.position = 0
        self.eta = eta
        self.chi = chi
        
    def put(self, transition, priority=1.0):
        self.buffer.append(transition)
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.buffer_limit

    def scale_priorities(self, td_errors):
        priorities = np.abs(td_errors.detach().numpy()) + self.chi
        priorities = np.power(priorities, self.eta)
        return priorities

    def normalize_priorities(self):
        total_priority = np.sum(self.priorities)
        if total_priority == 0.0:
            normalized_priorities = np.ones_like(self.priorities) / len(self.priorities)
        else:
            normalized_priorities = self.priorities / total_priority
        return normalized_priorities

    def update_priorities(self, td_errors):
        new_priorities = self.scale_priorities(td_errors).flatten()  # Flatten the priorities
        self.priorities[self.position - len(new_priorities):self.position] = new_priorities
        
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, numStates = 2, numActions=20, epsilon=0.98, min_epsilon=0.05, 
                 epsilon_decay=0.9999, gamma=0.98, lr = 0.0003, tau=0.005, 
                 batch_size = 32, wait_period = 500, grad_steps = 1, reward_norm = False, eta = 1.0):
        super(DQN, self).__init__()
        self.numStates = numStates
        self.numActions = numActions
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.eta = eta
        self.wait_period = wait_period
        self.grad_steps = grad_steps
        self.batch_size = batch_size
        self.reward_norm = reward_norm
        self.actionVec = np.linspace(-1, 1, numActions)
        self.q = Qnet(numStates,numActions)
        self.q_target = Qnet(numStates,numActions)
        self.q_target.load_state_dict(self.q.state_dict())
        self.memory = PrioritizedBuffer(50000, numStates, self.eta)
        self.tderror_history = []
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)
        self.eps = np.finfo(np.float32).eps.item()

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            
    def act(self, state):
        state = torch.tensor(state, dtype = torch.float)
        self.action = self.q.sample_action(state, self.epsilon)  
        return self.actionVec[self.action]
        
    def observe(self, state, action, reward, newState, done):
        #state = torch.tensor(state, dtype = torch.float)
        #newState = torch.tensor(newState, dtype = torch.float)
        #state = (state-self.state_mean)/(self.state_std + self.eps)
        #newState = (newState-self.state_mean)/(self.state_std + self.eps)
        done_mask = 0.0 if done else 1.0
        self.memory.put((state,self.action,reward,newState,done_mask))

    def train(self):
        if self.memory.size()>self.wait_period:
            self.train_net(self.q, self.q_target, self.memory, self.optimizer)

    def train_net(self, q, q_target, memory, optimizer):
        for i in range(self.grad_steps):
            s,a,r,s_prime,done_mask = self.memory.sample(self.batch_size)
            a = a.long()
            q_out = self.q(s)
            q_a = q_out.gather(1,a)
            max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
            
            td_errors = abs((r + self.gamma * max_q_prime * done_mask).detach() - q_a)
            self.memory.update_priorities(td_errors)
            self.tderror_history.append(np.nanmean(td_errors.detach().numpy()))
            
            target = r + self.gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 100.0)
            self.optimizer.step()
            self.update_epsilon()


