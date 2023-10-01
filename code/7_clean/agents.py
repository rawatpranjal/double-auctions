########################################################################
#                               AGENTS
########################################################################

import pandas as pd
import numpy as np

class Trader:
    def __init__(self, gameData, disclosure, index, buyer, type):
        self.gameTypes, self.numBuyers, self.numSellers, self.numTokens, self.numRounds, self.numPeriods, self.numSteps = gameData
        self.buyer = buyer
        self.index = index
        self.type = type
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
    def __init__(self, gameData, disclosure, index, buyer, type):
        super().__init__(gameData, disclosure, index, buyer, type)
    
    def bid(self):
        self.stepBid = self.stepTokenValue
        return self.stepBid
    
    def ask(self):
        self.stepAsk = self.stepTokenValue
        return self.stepAsk

class ZeroIntelligence(Trader):
    def __init__(self, gameData, disclosure, index, buyer, type):
        super().__init__(gameData, disclosure, index, buyer, type)
    
    def bid(self):
        self.stepBid = np.nan
        if self.stepTokenValue > 0:
            self.stepBid = np.round(np.random.uniform(self.stepTokenValue*0.1, self.stepTokenValue*1.0, 1).item(),1)
        return self.stepBid
        
    def ask(self):
        self.stepAsk = np.nan
        if self.stepTokenValue > 0:
            self.stepAsk = np.round(np.random.uniform(self.stepTokenValue, self.stepTokenValue*1.9, 1).item(),1)
        return self.stepAsk

def generateAgents(gameData,buyerStrategies,sellerStrategies,disclosure):
    buyers, sellers = [], []
    
    for idx,i in enumerate(buyerStrategies):
        if i == 'TruthTeller':
            buyers.append(TruthTeller(gameData, disclosure, index=idx, buyer=0, type=0)) 
        if i == 'ZeroIntelligence':
            buyers.append(ZeroIntelligence(gameData, disclosure, index=idx, buyer=0, type=1)) 
        if i == 'Reinforcer':
            buyers.append(Reinforcer(gameData, disclosure, index=idx, buyer=0, type=2)) 

    for idx,i in enumerate(sellerStrategies):
        if i == 'TruthTeller':
            sellers.append(TruthTeller(gameData, disclosure, index=idx, buyer=1, type=0)) 
        if i == 'ZeroIntelligence':
            sellers.append(ZeroIntelligence(gameData, disclosure, index=idx, buyer=1, type=1)) 
        if i == 'Reinforcer':
            sellers.append(Reinforcer(gameData, disclosure, index=idx, buyer=1, type=2)) 

    return buyers, sellers

def resetRounds(buyers, sellers, redemptionValues, tokenCosts):
    for i, buyer in enumerate(buyers):
        buyer.resetRound(redemptionValues[i,:])
    for i, seller in enumerate(sellers):
        seller.resetRound(tokenCosts[i,:])

def resetPeriods(buyers, sellers):
    for i,agent in enumerate(buyers + sellers):
        agent.resetPeriod()

def resetSteps(buyers, sellers):
    for i,agent in enumerate(buyers + sellers):
        agent.resetStep()

def collectOffers(buyers, sellers):
    bids, asks = [], []
    for i, buyer in enumerate(buyers):
        bids.append(buyer.bid())
    for i, seller in enumerate(sellers):
        asks.append(seller.ask())    
    return bids, asks

def bestOffers(bids, asks):
    if np.all(np.isnan(bids)) == False:
        currentBidIdx = int(np.nanargmax(bids))
        currentBid = np.nanmax(bids)
    else:
        currentBidIdx = np.nan
        currentBid = np.nan

    if np.all(np.isnan(asks)) == False:
        currentAskIdx = int(np.nanargmin(asks))
        currentAsk = np.nanmin(asks)
    else:
        currentAskIdx = np.nan
        currentAsk = np.nan
    return currentAsk, currentAskIdx, currentBid, currentBidIdx

def trade(buyers, sellers, currentAsk, currentAskIdx, currentBid, currentBidIdx, beta = 0.5):
        price, buy, sell = np.nan, np.nan, np.nan
        if (currentBidIdx >= 0) and (currentAskIdx >= 0):
            buy = buyers[currentBidIdx].buy(currentBid,currentAsk)
            sell = sellers[currentAskIdx].sell(currentBid,currentAsk)
            if buy and not sell:
                price = currentAsk
            elif sell and not buy:
                price = currentBid
            elif sell and buy: 
                price = beta*currentBid + (1-beta)*currentAsk
        return price, buy, sell

def observe(buyers, sellers, data):
    for i, agent in enumerate(buyers+sellers):
        agent.df = pd.concat([agent.df, data.to_frame().T], ignore_index=True)

def profit(value,price,buyer):
    if buyer == 0:
        return value - price
    else:
        return price - value
