from utils import *
from algorithms import * 

import pandas as pd
from copy import deepcopy

class Trader:
    def __init__(self, gameData, disclosure, index, buyer, reinforcer):
        self.gameType, self.numBuyers, self.numSellers, self.numTokens, self.numRounds, self.numPeriods, self.numSteps, self.seed = gameData
        self.index = index
        self.buyer = buyer
        self.reinforcer = reinforcer
        self.df = pd.DataFrame(columns=disclosure)
        self.disclosure = disclosure
        self.gameTokens = []
        self.gameTrades = 0
        self.gameProfits = 0
        self.gameRounds = 0
        self.gameSteps = 0
        self.roundTrades = 0
        self.roundProfits = 0
        self.roundPeriods = 0
        self.periodTrades = 0
        self.periodProfits = 0
        self.periodSteps = 0
        self.stepTrades = 0
        self.stepProfits = 0
        self.stepTokenValue = 0
        self.periodSteps = 0
        self.stepTrades = 0
        self.stepProfits = 0
        self.stepTokenValue = 0
        
    def startRound(self, tokenValues):
        self.roundTokens = tokenValues
        self.roundTrades = 0
        self.roundProfits = 0
        self.roundPeriods = 0

    def endRound(self):
        self.gameTokens.append(self.roundTokens)
        self.gameTrades += self.roundTrades
        self.gameProfits += self.roundProfits
        self.gameRounds += 1
           
    def startPeriod(self):
        self.periodTokens = self.roundTokens
        self.periodTrades = 0
        self.periodProfits = 0
        self.periodSteps = 0

    def endPeriod(self):
        self.roundProfits += self.periodProfits
        self.roundTrades += self.periodTrades
        self.roundPeriods += 1

    def startStep(self):
        self.stepProfits = 0
        self.stepTrades = 0
        self.stepTokenValue = np.nan
        if self.periodTrades < self.numTokens:
            self.stepTokenValue = self.periodTokens[self.periodTrades]
        if self.reinforcer == 1:
            self.state = generateState(self)

    def endStep(self):
        self.gameSteps += 1
        self.periodSteps += 1
        self.periodProfits += self.stepProfits
        self.periodTrades += self.stepTrades  

    def buy(self, currentBid, currentAsk):
        self.acceptSale = False
        if self.stepTokenValue >= currentAsk:
            self.acceptSale = True
        return self.acceptSale

    def sell(self, currentBid, currentAsk):
        self.acceptSale = False
        if self.stepTokenValue <= currentBid:
            self.acceptSale = True
        return self.acceptSale
    
    def transact(self, price):
        self.stepTrades = 1
        self.stepProfits = profit(self.stepTokenValue,price,self.buyer)

class TruthTeller(Trader):
    def __init__(self, gameData, disclosure, index, buyer, reinforcer):
        super().__init__(gameData, disclosure, index, buyer, reinforcer)
    
    def bid(self):
        self.stepBid = self.stepTokenValue
        return self.stepBid
    
    def ask(self):
        self.stepAsk = self.stepTokenValue
        return self.stepAsk

class ZeroIntelligence(Trader):
    def __init__(self, gameData, disclosure, index, buyer, reinforcer):
        super().__init__(gameData, disclosure, index, buyer, reinforcer)
    
    def bid(self):
        self.stepBid = np.nan
        if self.stepTokenValue >= 0:
            self.stepBid = np.random.uniform(self.stepTokenValue*0.1,self.stepTokenValue, 1).item()
            self.stepBid = np.round(self.stepBid, 1)
        return np.round(self.stepBid,1)
        
    def ask(self):
        self.stepAsk = np.nan
        if self.stepTokenValue >= 0:
            self.stepAsk = np.random.uniform(self.stepTokenValue,self.stepTokenValue*1.9, 1).item()
            self.stepAsk = np.round(self.stepAsk, 1)
        return self.stepAsk

def generateAgents(gameData,buyerStrategies,sellerStrategies,disclosure):
    buyers, sellers = [], []
    for idx,i in enumerate(buyerStrategies):
        if i == 'TruthTeller':
            buyers.append(TruthTeller(gameData, disclosure, index=idx, buyer=1, reinforer=0)) 
        if i == 'ZeroIntelligence':
            buyers.append(ZeroIntelligence(gameData, disclosure, index=idx, buyer=1, reinforer=0)) 
        if i == 'VPG':
            buyers.append(VPG(gameData, disclosure, index=idx, buyer=1, reinforcer=1, episodeLength = gameData[7])) 
        if i == 'PPO':
            buyers.append(PPO(gameData, disclosure, index=idx, buyer=1, reinforcer=1)) 
        if i == 'SAC':
            buyers.append(SAC(gameData, disclosure, index=idx, buyer=1, reinforcer=1)) 
        if i == 'DQN':
            buyers.append(DQN(gameData, disclosure, index=idx, buyer=1, reinforcer=1)) 
        if i == 'DDPG':
            buyers.append(DDPG(gameData, disclosure, index=idx, buyer=1, reinforcer=1)) 

    for idx,i in enumerate(sellerStrategies):
        if i == 'TruthTeller':
            sellers.append(TruthTeller(gameData, disclosure, index=idx, buyer=0, reinforcer=0)) 
        if i == 'ZeroIntelligence':
            sellers.append(ZeroIntelligence(gameData, disclosure, index=idx, buyer=0, reinforcer=0)) 
        if i == 'VPG':
            sellers.append(VPG(gameData, disclosure, index=idx, buyer=0, reinforcer=1)) 
        if i == 'PPO':
            sellers.append(PPO(gameData, disclosure, index=idx, buyer=0, reinforcer=1)) 
        if i == 'SAC':
            sellers.append(SAC(gameData, disclosure, index=idx, buyer=0, reinforcer=1)) 
        if i == 'DQN':
            sellers.append(DQN(gameData, disclosure, index=idx, buyer=0, reinforcer=1)) 
        if i == 'DDPG':
            sellers.append(DDPG(gameData, disclosure, index=idx, buyer=0, reinforcer=1)) 
    return buyers, sellers

class Reinforcer(Trader):
    def __init__(self, gameData, disclosure=[], index=0, buyer=1, reinforcer=1, numStates = 1, numActions=10, algo='BASE', depth = 0, verbose = 0, algoArgs=[]):
        super().__init__(gameData, disclosure, index, buyer, reinforcer)
        self.depth = depth
        self.disclosure = disclosure
        self.state = generateState(self)
        self.numStates = numStates
        self.numActions = numActions
        self.state = [0]*self.numStates
        self.algo = loadAlgo(algo, algoArgs)
        self.done = False
        self.verbose = verbose
    
    def observe(self):
        self.newState = generateState(self)
        self.algo.observe(self.state, self.action, self.stepProfits, self.newState, self.done)
        if self.verbose == 1:
            print(f'state:{self.state}, action:{self.action}, reward:{self.stepProfits}, newstate:{self.newState},done:{self.done}')
        self.state = self.newState

    def train(self):
        self.algo.train()
        
    def bid(self):
        self.stepBid = np.nan
        self.action = self.algo.act(self.state)
        if self.stepTokenValue >= 0:
            min = self.stepTokenValue * 0.01
            max = self.stepTokenValue * 1.5
            #frac = (F.tanh(torch.tensor(self.action[0])).item()+1)/2
            frac = (self.action+1)/2
            self.stepBid = min*(1-frac) + frac*max
        return self.stepBid
        
    def ask(self):
        self.stepAsk = np.nan
        self.action = self.algo.act(self.state)
        if self.stepTokenValue >= 0:
            frac = (self.action+1)/2
            min = self.stepTokenValue * 0.5
            max = self.stepTokenValue * 2.0
            #frac = (F.tanh(torch.tensor(self.action[0])).item()+1)/2
            frac = (self.action[0]+1)/2
            self.stepAsk = min * frac + (1-frac) * min
        return self.stepAsk