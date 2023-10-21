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
    if algo=='BASE':
        return BASE(*algoArgs)
    if algo=='QL':
        return QL(*algoArgs)
    if algo=='SARSA':
        return SARSA(*algoArgs)
    if algo=='VPG':
        return VPG(*algoArgs)
    if algo=='VPGContinuous':
        return VPGContinuous(*algoArgs)
    if algo=='DQN':
        return DQN(*algoArgs)
    if algo=='DQNPER':
        return DQNPER(*algoArgs)
    if algo=='A2C':
        return A2C(*algoArgs)
    if algo=='PPO1':
        return PPO1(*algoArgs)
    if algo=='PPO2':
        return PPO2(*algoArgs)
    if algo=='SAC':
        return SAC(*algoArgs) 
    if algo=='SAC2':
        return SAC2(*algoArgs)
    if algo=='DDPG':
        return DDPG(*algoArgs)  

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

class QL:
    def __init__(self, numStates, numActions=5, epsilon=0.99,
                 min_epsilon=0.05, epsilon_decay=0.9999, alpha=0.1, gamma=0.95):
        self.numActions = numActions
        self.numStates = 10
        self.Qtable = np.random.normal(0, 1, (10, 5, self.numActions))
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.actionVec = np.linspace(-1, 1, numActions)
        self.memory = []

    def act(self, state):
        state1, state2 = state
        if random.random() < self.epsilon:
            self.action = random.choice(range(self.numActions))
            return self.actionVec[self.action]
        else:
            self.action = np.argmax(self.Qtable[state1][state2])
            return self.actionVec[self.action]

    def observe(self, state, action, reward, newState, done):
        self.memory.append([state, self.action, reward / 100, newState, done])

    def train(self):
        state, action, reward, next_state, done = self.memory[-1]
        state1, state2 = state
        next_state1, next_state2 = next_state
        if done or (next_state1 == 10):
            td_target = reward
        else:
            best_action = np.argmax(self.Qtable[next_state1][next_state2])
            max_q_value = self.Qtable[next_state1][next_state2][best_action]
            td_target = reward + self.gamma * max_q_value
        td_error = td_target - self.Qtable[state1][state2][action]
        self.Qtable[state1][state2][action] += self.alpha * td_error
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

class SARSA:
    def __init__(self, numStates, numActions=5, epsilon=0.99,
                 min_epsilon=0.05, epsilon_decay=0.9999, alpha=0.1, gamma=0.95):
        self.numActions = numActions
        self.numStates = 10
        self.Qtable = np.random.normal(0, 1, (10, 5, numActions))
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.actionVec = np.linspace(-1, 1, numActions)
        self.memory = []
        self.action = None
        self.next_action = None

    def act(self, state):
        state1, state2 = state
        if random.random() < self.epsilon:
            self.action = random.choice(range(self.numActions))
            return self.actionVec[self.action]
        else:
            self.action = np.argmax(self.Qtable[state1][state2])
            return self.actionVec[self.action]
            
    def observe(self, state, action, reward, next_state, done):
        self.memory.append([state, self.action, reward / 100, next_state, done])

    def train(self):
        if len(self.memory)>=2:
            state, action, reward, next_state, done = self.memory[-1]
            next_state, next_action, next_reward, next_next_state, next_done = self.memory[-2]
    
            state1, state2 = state
            next_state1, next_state2 = next_state
    
            if done or (next_state1 == 10):
                td_target = reward
            else:
                td_target = reward + self.gamma * self.Qtable[next_state1][next_state2][next_action]
    
            td_error = td_target - self.Qtable[state1][state2][action]
            self.Qtable[state1][state2][action] += self.alpha * td_error
    
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

###################################################################################################
#                                   VPG
###################################################################################################

class VPG(nn.Module): # discrete
    def __init__(self, numStates, episodeLength = 10, numActions=5, numTrajectories = 1, lr = 0.0003,
                 epsilon=0.99, min_epsilon=0.01, gamma=0.99, epsilon_decay=0.995, reward_norm = False):
        super(VPG, self).__init__()
        self.memory = []
        self.rewards = []
        self.log_probs = []
        self.epsilon = epsilon
        self.lr = lr
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.numActions = numActions
        self.numStates = numStates
        self.gamma = gamma
        self.reward_norm = reward_norm
        self.numTrajectories = numTrajectories
        self.fc1 = nn.Linear(numStates, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, numActions)
        init.orthogonal_(self.fc1.weight, gain=0.01)
        init.orthogonal_(self.fc2.weight, gain=0.01)
        init.orthogonal_(self.fc3.weight, gain=0.01)
        self.episodeLength = episodeLength
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.done = False
        self.eps = np.finfo(np.float32).eps.item()
        self.actionVec = np.linspace(-1,1,numActions)
        self.policy_loss = [] 
        self.traj = 0 
        state_mean = np.loadtxt('mean.csv', delimiter=',')
        state_std = np.loadtxt('std.csv', delimiter=',')
        self.state_mean = torch.tensor(state_mean,dtype=torch.float)
        self.state_std = torch.tensor(state_std,dtype=torch.float)
        self.reward_mean, self.reward_std = np.loadtxt('mean_rewards.csv', delimiter=',')
        self.state_count = 0
        self.entropy = []

    def forward(self, x):
        x = x.float()
        x = (x - self.state_mean)/(self.state_std + self.eps)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def observe(self, state, action, reward, newState, done):
        if self.reward_norm:
            reward = ((reward - self.reward_mean)/(self.reward_std+self.eps)).item()
        self.memory.append([state, self.action.item(), reward, newState, done])
        self.rewards.append(reward)
        self.done = done

    def act(self, state):
        state = torch.tensor(state).unsqueeze(0).float()
        probs = self.forward(state)
        multinomial = Categorical(probs)
        if random.random() < self.epsilon:
            self.action = torch.randint(low=0, high=self.numActions - 1, size=(1,))
        else:
            self.action = multinomial.sample()
        self.log_probs.append(multinomial.log_prob(self.action))
        return self.actionVec[self.action.item()]
    
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
            self.update_epsilon()
            del self.rewards[:]
            del self.log_probs[:]
            self.traj += 1
            if self.traj == self.numTrajectories:
                self.optimizer.zero_grad()
                policy_loss = torch.cat(self.policy_loss).sum()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0)
                self.optimizer.step()
                self.policy_loss = []
                self.traj = 0

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

            
class VPGContinuous(nn.Module):
    def __init__(self, numStates=2, episodeLength=10, numTrajectories=5, gamma=0.99, 
                 std_init = 0.5, std_decay = 0.9995, std_min = 0.02, lr = 0.0003):
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
        state_mean = np.loadtxt('mean.csv', delimiter=',')
        state_std = np.loadtxt('std.csv', delimiter=',')
        self.state_mean = torch.tensor(state_mean, dtype=torch.float)[:self.numStates]
        self.state_std = torch.tensor(state_std, dtype=torch.float)[:self.numStates]
        self.state_count = 0
        self.std = std_init
        self.std_decay = std_decay
        self.std_min = std_min
        print(self.std , self.std_decay, self.std_min)

    def forward(self, x):
        x = x.float()
        x = (x - self.state_mean) / (self.state_std + self.eps)
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
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                self.update_std()
                self.policy_loss = []
                self.traj = 0

    def update_std(self):
        if self.std > self.std_min:
            self.std *= self.std_decay


###################################################################################################
#                                   A2C
###################################################################################################

class A2C(nn.Module):
    def __init__(self, numStates=2, numActions = 20, episodeLength=10, numTrajectories=5, gamma=0.99, max_train_steps = 60000,
                 lr = 0.0003, update_interval = 5):
        super(A2C, self).__init__()
        self.memory = []
        self.s_lst = []
        self.sp_lst = []
        self.a_lst = []
        self.r_lst = []
        self.mask_lst = []
        self.policy_loss = []
        self.numStates = numStates
        self.numActions = numActions
        self.episodeLength = episodeLength
        self.gamma = gamma
        self.numTrajectories = numTrajectories
        self.lr = lr
        self.traj = 0
        self.fc1 = nn.Linear(self.numStates, 128)
        self.fc2 = nn.Linear(self.numStates, 128)
        self.fc_pi = nn.Linear(128, self.numActions)
        self.fc_v = nn.Linear(128, 1)
        init.orthogonal_(self.fc1.weight, gain=0.01)
        init.orthogonal_(self.fc2.weight, gain=0.01)
        init.orthogonal_(self.fc_pi.weight, gain=0.01)
        init.orthogonal_(self.fc_v.weight, gain=0.01)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.step_idx = 0
        self.done = False
        self.eps = np.finfo(np.float32).eps.item()
        state_mean = np.loadtxt('mean.csv', delimiter=',')
        state_std = np.loadtxt('std.csv', delimiter=',')
        self.state_mean = torch.tensor(state_mean, dtype=torch.float)
        self.state_std = torch.tensor(state_std, dtype=torch.float)
        self.actionVec = np.linspace(-1,1,numActions)

    def pi(self, x, softmax_dim=1):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x)
        return prob

    def v(self, x):
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

    def act(self, state):
        state = torch.tensor(state, dtype = torch.float)
        self.state = (state - self.state_mean) / (self.state_std + self.eps)
        prob = self.pi(self.state)
        self.action = Categorical(prob).sample().numpy()
        return self.actionVec[self.action.item()]

    def observe(self, state, action, reward, newState, done):
        newState = torch.tensor(newState, dtype = torch.float)
        self.newState = (newState - self.state_mean) / (self.state_std + self.eps)
        self.memory.append([self.state, self.action.item(), reward, self.newState.tolist(), done])
        self.s_lst.append(self.state.tolist())
        self.sp_lst.append(self.newState.tolist())
        self.a_lst.append(self.action.item())
        self.r_lst.append(reward)
        self.mask_lst.append(1-done)
        self.step_idx += 1

    def train(self):
        if self.step_idx == (self.episodeLength-1):
            s_final = self.newState
            v_final = self.v(s_final).detach().clone().numpy()
            td_target = self.compute_target(v_final, self.r_lst, self.mask_lst)
            td_target_vec = td_target.reshape(-1)
            s_vec = torch.tensor(self.s_lst).float().reshape(-1, self.numStates)  # 4 == Dimension of state
            a_vec = torch.tensor(self.a_lst).reshape(-1).unsqueeze(1)
            advantage = td_target_vec - self.v(s_vec).reshape(-1)

            pi = self.pi(s_vec, softmax_dim=1)
            pi_a = pi.gather(1, a_vec).reshape(-1)
            loss = -(torch.log(pi_a) * advantage.detach()).mean() + F.smooth_l1_loss(self.v(s_vec).reshape(-1), td_target_vec)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            self.step_idx = 0

            del self.r_lst[:]
            del self.a_lst[:]
            del self.s_lst[:]
            del self.mask_lst[:]
            del self.sp_lst[:]

    def compute_target(self, v_final, r_lst, mask_lst):
        G = v_final.reshape(-1)
        td_target = []
        for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
            G = r + self.gamma * G * mask
            td_target.append(G)
        return torch.tensor(td_target[::-1]).float()

###################################################################################################
#                                   DQN, DDPG
###################################################################################################

class ReplayBuffer():
    def __init__(self, buffer_limit, numStates):
        self.buffer_limit = buffer_limit
        self.numStates = numStates
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)


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


class DQNPER(nn.Module):
    def __init__(self, numStates = 2, numActions=20, epsilon=0.98, min_epsilon=0.05, 
                 epsilon_decay=0.9999, gamma=0.98, lr = 0.0003, tau=0.005, 
                 batch_size = 32, wait_period = 500, grad_steps = 1, reward_norm = False, eta = 1.0):
        super(DQNPER, self).__init__()
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
        state_mean = np.loadtxt('mean.csv', delimiter=',')
        state_std = np.loadtxt('std.csv', delimiter=',')
        self.state_mean = torch.tensor(state_mean,dtype=torch.float)
        self.state_std = torch.tensor(state_std,dtype=torch.float)
        self.eps = np.finfo(np.float32).eps.item()
        self.reward_mean, self.reward_std = np.loadtxt('mean_rewards.csv', delimiter=',')

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            
    def act(self, state):
        state = torch.tensor(state, dtype = torch.float)
        state = (state-self.state_mean)/(self.state_std + self.eps)
        self.action = self.q.sample_action(state, self.epsilon)  
        return self.actionVec[self.action]
        
    def observe(self, state, action, reward, newState, done):
        if self.reward_norm:
            reward = ((reward - self.reward_mean)/(self.reward_std+self.eps)).item()
        state = torch.tensor(state, dtype = torch.float)
        newState = torch.tensor(newState, dtype = torch.float)
        state = (state-self.state_mean)/(self.state_std + self.eps)
        newState = (newState-self.state_mean)/(self.state_std + self.eps)
        done_mask = 0.0 if done else 1.0
        self.memory.put((state.tolist(),self.action,reward,newState.tolist(),done_mask))

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

class DQN(nn.Module):
    def __init__(self, numStates = 2, numActions=20, epsilon=0.98, min_epsilon=0.05, 
                 epsilon_decay=0.9999, gamma=0.98, lr = 0.0003, tau=0.005, 
                 batch_size = 32, wait_period = 500, grad_steps = 1, reward_norm = False):
        super(DQN, self).__init__()
        self.numStates = numStates
        self.numActions = numActions
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.wait_period = wait_period
        self.grad_steps = grad_steps
        self.batch_size = batch_size
        self.reward_norm = reward_norm
        self.actionVec = np.linspace(-1, 1, numActions)
        self.q = Qnet(numStates,numActions)
        self.q_target = Qnet(numStates,numActions)
        self.q_target.load_state_dict(self.q.state_dict())
        self.memory = ReplayBuffer(50000, numStates)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)
        state_mean = np.loadtxt('mean.csv', delimiter=',')
        state_std = np.loadtxt('std.csv', delimiter=',')
        self.state_mean = torch.tensor(state_mean,dtype=torch.float)
        self.state_std = torch.tensor(state_std,dtype=torch.float)
        self.eps = np.finfo(np.float32).eps.item()
        self.reward_mean, self.reward_std = np.loadtxt('mean_rewards.csv', delimiter=',')

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            
    def act(self, state):
        state = torch.tensor(state, dtype = torch.float)
        state = (state-self.state_mean)/(self.state_std + self.eps)
        self.action = self.q.sample_action(state, self.epsilon)  
        return self.actionVec[self.action]
        
    def observe(self, state, action, reward, newState, done):
        if self.reward_norm:
            reward = ((reward - self.reward_mean)/(self.reward_std+self.eps)).item()
        state = torch.tensor(state, dtype = torch.float)
        newState = torch.tensor(newState, dtype = torch.float)
        state = (state-self.state_mean)/(self.state_std + self.eps)
        newState = (newState-self.state_mean)/(self.state_std + self.eps)
        done_mask = 0.0 if done else 1.0
        self.memory.put((state.tolist(),self.action,reward,newState.tolist(),done_mask))

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
            target = r + self.gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
            self.optimizer.step()
            self.update_epsilon()


class MuNet(nn.Module):
    def __init__(self, numStates):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(numStates, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)
        init.orthogonal_(self.fc1.weight, gain=0.01)
        init.orthogonal_(self.fc2.weight, gain=0.01)
        init.orthogonal_(self.fc_mu.weight, gain=0.0001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu

class QNet2(nn.Module):
    def __init__(self, numStates):
        super(QNet2, self).__init__()
        self.fc_s = nn.Linear(numStates, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)
        init.orthogonal_(self.fc_s.weight, gain=0.01)
        init.orthogonal_(self.fc_a.weight, gain=0.01)
        init.orthogonal_(self.fc_q.weight, gain=0.01)
        init.orthogonal_(self.fc_out.weight, gain=0.0001)

    def forward(self, x, a):
        h1 = F.leaky_relu(self.fc_s(x))
        h2 = F.leaky_relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.leaky_relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.01
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class DDPG(nn.Module):
    def __init__(self, numStates = 2, gamma=0.99, lr_mu = 0.0003, lr_q = 0.0003, tau=0.005, 
                 batch_size = 32, buffer_limit = 50000, wait_period = 2000, grad_steps = 1,
                 reward_norm = False, std_init = 0.5, std_decay = 0.9995, std_min = 0.02):
        super(DDPG, self).__init__()
        self.numStates = numStates
        self.gamma = gamma
        self.lr_mu = lr_mu
        self.lr_q = lr_q
        self.buffer_limit = buffer_limit
        self.tau = tau
        self.wait_period = wait_period
        self.grad_steps = grad_steps
        self.batch_size = batch_size
        self.reward_norm = reward_norm
        state_mean = np.loadtxt('mean.csv', delimiter=',')
        state_std = np.loadtxt('std.csv', delimiter=',')
        self.state_mean = torch.tensor(state_mean,dtype=torch.float)
        self.state_std = torch.tensor(state_std,dtype=torch.float)
        self.eps = np.finfo(np.float32).eps.item()
        self.reward_mean, self.reward_std = np.loadtxt('mean_rewards.csv', delimiter=',')
        self.memory = ReplayBuffer(buffer_limit, numStates)
        self.q, self.q_target = QNet2(numStates), QNet2(numStates)
        self.q_target.load_state_dict(self.q.state_dict())
        self.mu, self.mu_target = MuNet(numStates), MuNet(numStates)
        self.mu_target.load_state_dict(self.mu.state_dict())
        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=self.lr_mu)
        self.q_optimizer  = optim.Adam(self.q.parameters(), lr=self.lr_q)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
        self.std = std_init
        self.std_decay = std_decay
        self.std_min = std_min
        self.cnt = 0
    
    def act(self, state):
        state = torch.tensor(state, dtype = torch.float)
        state = (state-self.state_mean)/(self.state_std + self.eps)
        if self.cnt<=self.wait_period:
            self.action = torch.tensor(np.random.uniform(-1,1,1))
            self.cnt += 1
            self.mean = self.action
        else:
            self.mean = self.mu(state) 
            #self.noise_value = self.ou_noise()[0]
            #self.action = torch.tanh(torch.tensor(self.mean.item() + self.noise_value, dtype = torch.float))
            dist = Normal(self.mean, self.std)
            self.action = torch.tanh(dist.sample())
        return self.action.item()
        
    def observe(self, state, action, reward, newState, done):
        if self.reward_norm:
            reward = ((reward - self.reward_mean)/(self.reward_std+self.eps)).item()
        state = torch.tensor(state, dtype = torch.float)
        newState = torch.tensor(newState, dtype = torch.float)
        state = (state-self.state_mean)/(self.state_std + self.eps)
        newState = (newState-self.state_mean)/(self.state_std + self.eps)
        done_mask = 0.0 if done else 1.0
        self.memory.put((state.tolist(),self.action,reward,newState.tolist(),done_mask))

    def train(self):
        if self.memory.size()>self.wait_period:
            self.train_net()
            self.soft_update(self.mu, self.mu_target)
            self.soft_update(self.q, self.q_target)
            
    def train_net(self):
        for i in range(self.grad_steps):  
            s,a,r,s_prime,done_mask  = self.memory.sample(self.batch_size)
            #a = a.long()
            target = r + self.gamma * self.q_target(s_prime, self.mu_target(s_prime)) * done_mask
            q_loss = F.smooth_l1_loss(self.q(s,a), target.detach())
            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
            self.q_optimizer.step()
            mu_loss = -self.q(s,self.mu(s)).mean() # That's all for the policy loss.
            self.mu_optimizer.zero_grad()
            mu_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mu.parameters(), 1.0)
            self.mu_optimizer.step()  
            self.update_std()

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        
    def update_std(self):
        if self.std > self.std_min:
            self.std *= self.std_decay



###################################################################################################
#                                   PPO
###################################################################################################


class PPO1(nn.Module):
    def __init__(self, num_states=2, learning_rate=0.0003, gamma=0.9, lmbda=0.9, eps_clip=0.1, K_epoch=10, rollout_len=1, buffer_size=10, minibatch_size=32, verbose=0):
        super(PPO1, self).__init__()
        self.data           = []
        self.rollout        = []
        self.num_states = num_states
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.rollout_len = rollout_len
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.verbose = -1
        self.fc1   = nn.Linear(num_states,64)
        self.fc2   = nn.Linear(num_states,64)
        self.fc3   = nn.Linear(num_states,64)
        self.fc_mu = nn.Linear(64,1)
        self.fc_std  = nn.Linear(64,1)
        self.fc_v = nn.Linear(64,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0
        self.score          = 0.0
        self.count          = 0
        self.seed           = 42
        self.done           = False
        self.verbose = verbose
        self.mus = []
        self.stds = []
        self.ass = []
        self.logps = []
        self.rs = []
        self.dones = []
        
    def act(self, s):
        s = torch.tensor(s).float()
        mu, std = self.pi(s)
        dist = Normal(mu, std)
        self.a = dist.sample()
        self.log_prob = dist.log_prob(self.a) 
        self.mus.append(mu.item())
        self.stds.append(std.item())
        self.ass.append(self.a.item())
        self.logps.append(self.log_prob.item())
        if self.verbose > 0:
            if self.count % self.verbose == 0:
                print('internal', mu.item(), std.item(), self.a.item(), self.log_prob.item())
        return self.a.item()
        
    def observe(self, s, a, r, s_prime, done):
        self.done = done
        self.rollout.append((s, self.a, r, s_prime, self.log_prob.item(), done))
        if len(self.rollout) == self.rollout_len:
            self.put_data(self.rollout)
            self.rollout = []
        self.count += 1
        self.score += r

    def pi(self, x, softmax_dim = 0):
        mu = F.leaky_relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(mu))
        std = F.leaky_relu(self.fc2(x))
        std = F.softplus(self.fc_std(std))
        return mu, std
    
    def v(self, x):
        v = F.leaky_relu(self.fc3(x))
        v = self.fc_v(v)
        return v
      
    def put_data(self, transition):
        s, a, r, s_prime, log_prob, done = transition[0]
        if self.verbose>0:
            if self.count % self.verbose == 0:
                print('ext', s, a.item(), r, s_prime, log_prob, done)
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


import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

class PPO2(nn.Module):
    def __init__(self, num_states=2, learning_rate=0.0003, gamma=0.9, lmbda=0.9, eps_clip=0.1, K_epoch=10, rollout_len=1, buffer_size=10, minibatch_size=4, verbose=0):
        super(PPO2, self).__init__()
        self.data           = []
        self.rollout        = []
        self.num_states = num_states
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.rollout_len = rollout_len
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.verbose = 10
        self.seed = 42
        self.fc1   = nn.Linear(num_states,64)
        self.fc1_h = nn.Linear(64,64)
        self.fc2   = nn.Linear(num_states,64)
        self.fc2_h = nn.Linear(64,64)
        self.fc3   = nn.Linear(num_states,256)
        self.fc3_h   = nn.Linear(num_states,256)

        self.fc_mu = nn.Linear(128,1)
        self.fc_std  = nn.Linear(128,1)
        self.fc_v = nn.Linear(256,1)
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc_mu.weight, gain=0.01)
        nn.init.orthogonal_(self.fc_std.weight, gain=0.01)
        nn.init.orthogonal_(self.fc_v.weight, gain=1)
        initial_learning_rate = 2.5e-4
        total_timesteps = 1e5
        self.lr_lambda = lambda epoch: 1 - (epoch / total_timesteps)
        self.optimizer = optim.Adam(self.parameters(), lr=initial_learning_rate, eps=1e-5, betas=(0.9, 0.999))
        self.scheduler = LambdaLR(self.optimizer, self.lr_lambda)
        self.optimization_step = 0
        self.std_min = 0.05
        self.max_std = 0.8
        self.score          = 0.0
        self.count          = 0
        self.eta            = 0.0
        self.seed           = 42
        self.done           = False
        self.verbose = verbose
        self.entropy_coeff = 0.01
        self.running_mean = torch.zeros(num_states)
        self.running_std = torch.ones(num_states) #* 1e-8

    def act(self, s):
        s = torch.from_numpy(np.array(s)).float()
        if self.count > 10:
            s = (s - self.running_mean)/self.running_std
        mu, std = self.pi(s)
        #print(mu, std, s, self.running_mean, self.running_std)
        std = torch.clamp(std, self.std_min, self.max_std)
        dist = Normal(mu, std)
        if self.verbose > 0:
            if self.count%self.verbose == 0:
                print('\nmu',mu.item(),'std',std.item())
        a = dist.sample()
        self.a = torch.tanh(a)
        self.log_prob = -0.5 * ((self.a - mu) / std)**2 - 0.5 * np.log(2 * np.pi) - torch.log(std)
        return self.a.item()
        
    def observe(self, s, a, r, s_prime, done):
        self.done = done
        self.rollout.append((s, self.a, r, s_prime, self.log_prob.item(), done))
        if len(self.rollout) == self.rollout_len:
            self.put_data(self.rollout)
            self.rollout = []
        self.count += 1
        self.score += r
        s = torch.from_numpy(np.array(s)).float()
        self.running_mean = (1 - 1 / self.count) * self.running_mean + (1 / self.count) * s
        self.running_std = (1 - 1 / self.count) * self.running_std + (1 / self.count) * (s - self.running_mean)
        
    def pi(self, x, softmax_dim = 0):
        mu = F.leaky_relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(mu))
        std = F.leaky_relu(self.fc2(x))
        std = torch.clamp(self.fc_std(std), min=-10, max=10)
        std = F.softplus(std)
        return mu, std
    
    def v(self, x):
        v = F.leaky_relu(self.fc3(x))
        v = self.fc_v(v)
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
                    std = torch.clamp(std, self.std_min, self.max_std)
                    #dist = Normal(mu, std)
                    #log_prob = dist.log_prob(a)
                    log_prob = -0.5 * ((a - mu) / std)**2 - 0.5 * np.log(2 * np.pi) - torch.log(std)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                    entropy_bonus = self.entropy_coeff * (-(torch.log(std) + 0.5 * np.log(2 * np.pi * np.e))).sum()
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target) - entropy_bonus
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1
            self.scheduler.step()


class PPO3(nn.Module):
    def __init__(self, num_states=8, learning_rate=0.0003, gamma=0.9, lmbda=0.9, eps_clip=0.2, K_epoch=10, rollout_len=1, buffer_size=10, minibatch_size=32, verbose=0):
        super(PPO3, self).__init__()
        self.data           = []
        self.rollout        = []
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
        self.fc2   = nn.Linear(num_states,128)
        self.fc3   = nn.Linear(num_states,128)
        self.fc_mu = nn.Linear(128,1)
        self.fc_std  = nn.Linear(128,1)
        self.fc_v = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0
        self.std_min = 0.02
        self.max_std = 0.5
        self.score          = 0.0
        self.count          = 0
        self.eta            = 0.0
        self.seed           = 42
        self.done           = False
        self.verbose = verbose
        self.entropy_coeff = 0.01
        self.running_reward_mean = 0.0
        self.running_reward_std = 1.0

    def act(self, s):
        mu, std = self.pi(torch.from_numpy(np.array(s)).float())
        std = torch.clamp(std, self.std_min, self.max_std)
        dist = Normal(mu, std)
        if self.verbose == 1:
            print('\nmu',mu.item(),'std',std.item())
        a = dist.sample()
        self.a = torch.tanh(a)
        self.log_prob = -0.5 * ((self.a - mu) / std)**2 - 0.5 * np.log(2 * np.pi) - torch.log(std)
        return self.a.item()
        
    def observe(self, s, a, r, s_prime, done):
        self.done = done
        self.rollout.append((s, self.a, r, s_prime, self.log_prob.item(), done))
        if len(self.rollout) == self.rollout_len:
            self.put_data(self.rollout)
            self.rollout = []
        self.count += 1
        self.score += r
        
    def pi(self, x, softmax_dim = 0):
        mu = F.leaky_relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(mu))
        std = F.leaky_relu(self.fc2(x))
        std = F.softplus(self.fc_std(std))
        return mu, std
    
    def v(self, x):
        v = F.leaky_relu(self.fc3(x))
        v = self.fc_v(v)
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
                    std = torch.clamp(std, self.std_min, self.max_std)
                    #dist = Normal(mu, std)
                    #log_prob = dist.log_prob(a)
                    log_prob = -0.5 * ((a - mu) / std)**2 - 0.5 * np.log(2 * np.pi) - torch.log(std)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                    entropy_bonus = self.entropy_coeff * (-(torch.log(std) + 0.5 * np.log(2 * np.pi * np.e))).sum()
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target) - entropy_bonus
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1



###################################################################################################
#                                   SAC
###################################################################################################

class ReplayBuffer_SAC():
    def __init__(self, buffer_limit, numStates):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class PolicyNet_SAC(nn.Module):
    def __init__(self, numStates, learning_rate, init_alpha, lr_alpha, target_entropy):
        super(PolicyNet_SAC, self).__init__()
        self.fc1 = nn.Linear(numStates, 128)
        self.fc_mu = nn.Linear(128,1)
        self.fc_std  = nn.Linear(128,1)
        init.orthogonal_(self.fc1.weight, gain=0.01)
        init.orthogonal_(self.fc_mu.weight, gain=0.01)
        init.orthogonal_(self.fc_std.weight, gain=0.01)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = target_entropy

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        torch.clamp(std, 0.02, 0.3)
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet_SAC(nn.Module):
    def __init__(self, numStates, learning_rate, tau):
        super(QNet_SAC, self).__init__()
        self.fc_s = nn.Linear(numStates, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_cat = nn.Linear(128,32)
        self.fc_out = nn.Linear(32,1)
        init.orthogonal_(self.fc_s.weight, gain=0.01)
        init.orthogonal_(self.fc_a.weight, gain=0.01)
        init.orthogonal_(self.fc_cat.weight, gain=0.01)
        init.orthogonal_(self.fc_out.weight, gain=0.01)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.tau = tau

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

class SAC(nn.Module):
    def __init__(self, numStates=2, lr_pi=0.0003, lr_q=0.0003, gamma=0.99, batch_size=64, buffer_limit = 50000,
                 target_entropy = -1.0, init_alpha = 0.01, lr_alpha = 0.0003, wait_period = 100, grad_steps = 1, tau = 0.005):
        super(SAC, self).__init__()
        self.numStates = numStates
        self.lr_pi = lr_pi
        self.lr_q = lr_q
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_limit = buffer_limit
        self.target_entropy = target_entropy
        self.init_alpha = init_alpha
        self.lr_alpha = lr_alpha
        self.wait_period = wait_period
        self.grad_steps = grad_steps
        self.tau = tau
        state_mean = np.loadtxt('mean.csv', delimiter=',')
        state_std = np.loadtxt('std.csv', delimiter=',')
        self.state_mean = torch.tensor(state_mean, dtype=torch.float)
        self.state_std = torch.tensor(state_std, dtype=torch.float)
        self.memory = ReplayBuffer(buffer_limit, numStates)
        self.q1, self.q2 = QNet_SAC(numStates, lr_q, tau), QNet_SAC(numStates, lr_q, tau),
        self.q1_target, self.q2_target =  QNet_SAC(numStates, lr_q, tau), QNet_SAC(numStates, lr_q, tau)
        self.pi = PolicyNet_SAC(numStates, lr_pi, init_alpha, lr_alpha, target_entropy)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.eps = 1e-8

    def act(self, state):
        state = torch.tensor(state, dtype = torch.float)
        state = (state - self.state_mean) / (self.state_std + self.eps)
        self.action, self.log_prob = self.pi(state)
        return self.action.item()
        
    def observe(self, state, action, reward, newState, done):
        state = torch.tensor(state, dtype = torch.float)
        state = (state - self.state_mean) / (self.state_std + self.eps)
        newState = torch.tensor(newState, dtype = torch.float)
        newState = (newState - self.state_mean) / (self.state_std + self.eps)
        self.memory.put((state.tolist(), self.action.item(), reward, newState.tolist(), done))

    def train(self):
        if self.memory.size()>self.wait_period:
            for i in range(self.grad_steps):
                mini_batch = self.memory.sample(self.batch_size)
                td_target = self.calc_target(self.pi, self.q1_target, self.q2_target, mini_batch)
                self.q1.train_net(td_target, mini_batch)
                self.q2.train_net(td_target, mini_batch)
                entropy = self.pi.train_net(self.q1, self.q2, mini_batch)
                self.q1.soft_update(self.q1_target)
                self.q2.soft_update(self.q2_target)

    def calc_target(self, pi, q1, q2, mini_batch):
        s, a, r, s_prime, done = mini_batch
    
        with torch.no_grad():
            a_prime, log_prob = pi(s_prime)
            entropy = -pi.log_alpha.exp() * log_prob
            q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = r + self.gamma * done * (min_q + entropy)
        return target
