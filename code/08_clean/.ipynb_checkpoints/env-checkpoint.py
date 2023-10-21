from utils import *
from agents import *
from algorithms import *
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces

class GymEnv(gym.Env):
    def __init__(self, metaData, buyers, sellers, log):
        self.gameData = metaData[0:8]
        self.gameTypes, self.numBuyers, self.numSellers, self.numTokens, self.numRounds, self.numPeriods, self.numSteps, self.seed = self.gameData
        self.disclosure, self.buyers, self.sellers = metaData[8:]
        self.log = log
        self.rnd = 0
        self.period = 0
        self.Step = 0
        self.buyers = buyers
        self.sellers = sellers
        self.action_space = spaces.Box(-1,1,(1,),dtype=np.float32)
        self.numStates = len(generateState(self.buyers[0]))
        self.observation_space = spaces.Box(-1,9,(self.numStates,),dtype=np.float32)
        startRounds(self.gameData, self.log, self.buyers, self.sellers, self.rnd)
    
    def reset(self, seed = None):
        startPeriods(self.buyers, self.sellers)
        self.buyers[0].state = generateState(self.buyers[0])
        return self.buyers[0].state, {}

    def step(self, action):
        # start step, update token values and generate state
        startSteps(self.buyers, self.sellers)

        # trade
        bids, asks = collectOffers(self.buyers, self.sellers)
        min = self.buyers[0].stepTokenValue*0.01
        max = self.buyers[0].stepTokenValue*1.5
        frac = (action+1)/2
        bids[0] = min * (1-frac) + frac * max

        # transact
        currentAsk, currentAskIdx, currentBid, currentBidIdx = bestOffers(bids, asks)
        price, buy, sell = trade(self.buyers, self.sellers, currentAsk, currentAskIdx, currentBid, currentBidIdx)

        # obtain profit / rewards
        bprofit, sprofit = 0, 0
        if price > 0:
            self.buyers[currentBidIdx].transact(price)
            self.sellers[currentAskIdx].transact(price)
            bprofit = self.buyers[currentBidIdx].stepProfits
            sprofit = self.sellers[currentAskIdx].stepProfits

        # update log, disclose information and update states
        self.log.addStep([self.rnd, self.period, self.Step, bids, asks, currentBid, currentBidIdx, currentAsk, currentAskIdx, buy, sell, price, price>0, bprofit, sprofit])
        observe(self.buyers, self.sellers, self.log.disclose())
        agentsObserve(self.buyers, self.sellers)

        # compute reward, newState, done
        newState = self.buyers[0].state
        done = self.buyers[0].done
        reward = 0.0
        if price > 0 and currentBidIdx == 0:
            reward = np.nan_to_num(bprofit,nan=0)

        # train agent
        agentsTrain(self.buyers, self.sellers)

        # end step
        endSteps(self.buyers, self.sellers)

        # if done with episode, end period
        self.Step += 1
        if done:
            endPeriods(self.buyers, self.sellers)
            self.period += 1
            self.Step = 0
        return newState, reward, done, False, {}