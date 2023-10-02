from setup import *
import pandas as pd

class Trader:
    def __init__(self, gameData, disclosure, index, buyer, reinforcer):
        self.gameType, self.numBuyers, self.numSellers, self.numTokens, self.numRounds, self.numPeriods, self.numSteps, self.seed = gameData
        self.index = index
        self.buyer = buyer
        self.reinforcer = reinforcer
        self.df = pd.DataFrame(columns=disclosure)
        self.disclosure = disclosure
        self.periodStep = 0
        self.roundPeriods = 0

    def resetRound(self, tokenValues):
        self.roundTokens = tokenValues
        self.roundTrades = 0
        self.roundProfits = 0
        self.roundPeriods = 0 
        
    def resetPeriod(self):
        self.periodTokens = self.roundTokens
        self.periodTrades = 0
        self.periodProfits = 0
        self.periodStep = 0

    def resetStep(self):
        self.stepProfits = 0
        self.stepTrades = 0
        self.stepTokenValue = np.nan
        self.stepBid = np.nan
        self.stepAsk = np.nan
        if self.periodTrades < self.numTokens:
            self.stepTokenValue = self.periodTokens[self.periodTrades]
        
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
        self.periodTrades += self.stepTrades
        self.periodProfits += self.stepProfits

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
            self.stepBid = np.random.uniform(self.stepTokenValue*0.5,self.stepTokenValue, 1).item()
            self.stepBid = np.round(self.stepBid, 1)
        return np.round(self.stepBid,1)
        
    def ask(self):
        self.stepAsk = np.nan
        if self.stepTokenValue >= 0:
            self.stepAsk = np.random.uniform(self.stepTokenValue,self.stepTokenValue*1.5, 1).item()
            self.stepAsk = np.round(self.stepAsk, 1)
        return self.stepAsk

def generateAgents(gameData,buyerStrategies,sellerStrategies,disclosure):
    buyers, sellers = [], []
    for idx,i in enumerate(buyerStrategies):
        if i == 'TruthTeller':
            buyers.append(TruthTeller(gameData, disclosure, index=idx, buyer=1, reinforer=0)) 
        if i == 'ZeroIntelligence':
            buyers.append(ZeroIntelligence(gameData, disclosure, index=idx, buyer=1, reinforer=0)) 
        if i == 'REINFORCE':
            buyers.append(REINFORCE(gameData, disclosure, index=idx, buyer=1, reinforcer=1)) 
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
        if i == 'REINFORCE':
            sellers.append(REINFORCE(gameData, disclosure, index=idx, buyer=0, reinforcer=1)) 
        if i == 'PPO':
            sellers.append(PPO(gameData, disclosure, index=idx, buyer=0, reinforcer=1)) 
        if i == 'SAC':
            sellers.append(SAC(gameData, disclosure, index=idx, buyer=0, reinforcer=1)) 
        if i == 'DQN':
            sellers.append(DQN(gameData, disclosure, index=idx, buyer=0, reinforcer=1)) 
        if i == 'DDPG':
            sellers.append(DDPG(gameData, disclosure, index=idx, buyer=0, reinforcer=1)) 
    return buyers, sellers
