########################################################################
#                               GAME SETUP
########################################################################

import numpy as np

def tokens(gameType='5555', numBuyers=4, numSellers=4, numTokens=4):
    # Convert game code into upper bounds for token values
    R1 = int(gameType[0])
    R2 = int(gameType[1])
    R3 = int(gameType[2])
    R4 = int(gameType[3])

    # Generate values
    A = np.random.uniform(0, R1, (numBuyers + numSellers, numTokens))  # Baseline randomness
    B = np.random.uniform(0, R2, (numBuyers + numSellers, 1))  # Agent differentiator
    C = np.random.uniform(0, R3, (1, numTokens))  # Token differentiator
    D = np.random.uniform(0, R4, (numBuyers, numTokens))  # Buyer-seller differentiator
    E = np.zeros((numSellers, numTokens))

    # Collect and normalize between 0-100
    tokenValues = A + B + C + np.r_[D, E]
    tokenValues = ((tokenValues - tokenValues.min()) / (tokenValues.max() - tokenValues.min())) * 100

    # Buyer valuations sort
    redemptionValues = tokenValues[0:numBuyers, :]
    sortedIndices = np.argsort(redemptionValues, axis=1)[:, ::-1]
    redemptionValues = np.take_along_axis(redemptionValues, sortedIndices, axis=1)

    # Seller costs sort
    tokenCosts = tokenValues[numBuyers:(numBuyers + numSellers), :]
    sortedIndicesCosts = np.argsort(tokenCosts, axis=1)
    tokenCosts = np.take_along_axis(tokenCosts, sortedIndicesCosts, axis=1)

    return np.round(redemptionValues,1), np.round(tokenCosts,1)

def schedules(redemptionValues, tokenCosts):
    prices = np.linspace(0, 100, 100)
    demand = np.zeros((100), dtype='int')
    supply = np.zeros((100), dtype='int')
    for i, p in enumerate(prices):
        demand[i] = np.sum(p <= redemptionValues)
        supply[i] = np.sum(p >= tokenCosts)
    return demand, supply, prices

def equilibrium(demand,supply,prices):
    peq, qeq = [], np.nan
    for i, p in enumerate(prices):
        if demand[i] == supply[i]:
            peq.append(p)
            qeq = demand[i]
    if len(peq) == 0:
        i = np.argmin(np.abs(demand-supply))
        peq = prices[i] 
        qeq = demand[i]
    return np.round(np.nanmean(peq),1), np.round(qeq,1)
    
def reservePrices(demand, supply,prices):
    arr = np.zeros_like(demand)
    change = np.where(demand[:-1] != demand[1:])[0]
    arr[change] = 1
    arr[-1] = 1
    buyerReservationPrices = prices[arr==1]
    arr = np.zeros_like(supply)
    change = np.where(supply[:-1] != supply[1:])[0]
    arr[change] = 1
    arr[-1] = 1
    sellerReservationPrices = prices[arr==1]
    return buyerReservationPrices, sellerReservationPrices
    
def surplus(redemptionValues,tokenCosts, peq, qeq):
    buyerSurplus = np.round(np.sum(redemptionValues-peq*np.where(redemptionValues-peq>=0,1,0)),1)
    sellerSurplus = np.round(np.sum(peq-tokenCosts*np.where(peq-tokenCosts>=0,1,0)),1)
    totalSurplus = np.round(buyerSurplus + sellerSurplus,1)
    buyerSurplusFrac = np.round(buyerSurplus/totalSurplus,2)
    sellerSurplusFrac = np.round(sellerSurplus/totalSurplus,2)
    return buyerSurplus, sellerSurplus, totalSurplus, buyerSurplusFrac, sellerSurplusFrac

def roundSetup(gameTypes, numBuyers, numSellers, numTokens, numRounds, numPeriods, numSteps):
    metadata = []
    redemptionValues, tokenCosts = tokens(gameTypes, numBuyers, numSellers, numTokens)
    demand, supply, prices = schedules(redemptionValues,tokenCosts)
    peq, qeq = equilibrium(demand,supply,prices)
    buyerReservationPrices, sellerReservationPrices = reservePrices(demand, supply,prices)
    buyerSurplus, sellerSurplus, totalSurplus, buyerSurplusFrac, sellerSurplusFrac = surplus(redemptionValues,tokenCosts, peq, qeq)
    metadata += [redemptionValues, tokenCosts, demand, supply, prices, peq, qeq]
    metadata += [buyerReservationPrices, sellerReservationPrices, buyerSurplus, sellerSurplus, totalSurplus, buyerSurplusFrac, sellerSurplusFrac]
    return metadata

########################################################################
#                                AGENTS
########################################################################
import pandas as pd

def profit(value,price,buyer):
    if buyer == 0:
        return value - price
    else:
        return price - value

class Trader:
    def __init__(self, gameData, disclosure, index, buyer, type):
        self.gameTypes, self.numBuyers, self.numSellers, self.numTokens, self.numRounds, self.numPeriods, self.numSteps = gameData
        self.buyer = buyer
        self.index = index
        self.type = type
        self.df = pd.DataFrame(columns=disclosure)
        self.disclosure = disclosure

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

########################################################################
#                                Log
########################################################################

class Log:
    def __init__(self, gameData, buyerStrategies, sellerStrategies,disclosure):
        self.gameTypes, self.numBuyers, self.numSellers, self.numTokens, self.numRounds, self.numPeriods, self.numSteps = gameData
        self.stepData = pd.DataFrame(columns=['rnd', 'period', 'step', 'bids','asks','currentBid','currentBidIdx','currentAsk',
                                             'currentAskIdx','buy','sell','price','sale', 'bprofit', 'sprofit'])
        self.roundData = pd.DataFrame(columns=['rnd', 'redemptionValues', 'tokenCosts', 'demand', 'supply', 'prices', 'peq', 'qeq', 
                                              'buyerReservationPrices', 'sellerReservationPrices', 'buyerSurplus', 'sellerSurplus', 
                                              'totalSurplus', 'buyerSurplusFrac', 'sellerSurplusFrac'])
        self.disclosure = disclosure
        
    def addStep(self, stepData):
        self.stepData.loc[len(self.stepData.index)] = stepData

    def addRound(self, roundData):
        self.roundData.loc[len(self.roundData.index)] = roundData
    
    def disclose(self):
        return self.stepData[self.disclosure].iloc[-1]
        
    def getPeriod(self, rnd, period):
        return self.stepData[(self.stepData.rnd==rnd) & (self.stepData.period==period)]
        
    def getPeriodActivity(self, rnd, period):
        periodData = self.getPeriod(rnd, period)
        periodBids = list(periodData.bids)
        periodAsks = list(periodData.asks)
        periodPrices = list(periodData.price)
        periodSales = np.cumsum(np.where(periodData.price > 0,1,0))
        return periodBids, periodAsks, periodPrices, periodSales

    def getRound(self, rnd):
        return self.roundData[(self.roundData.rnd==rnd)]

    def getRoundList(self, rnd):
        return self.getRound(rnd).values.tolist()[0][1:]

########################################################################
#                                Trading Env
########################################################################

class TradingGameEnv:
    def __init__(self, metaData):
        self.gameData = metaData[0:7]
        self.gameTypes, self.numBuyers, self.numSellers, self.numTokens, self.numRounds, self.numPeriods, self.numSteps = self.gameData
        self.disclosure, self.buyerStrategies, self.sellerStrategies = metaData[7:]
        self.buyers, self.sellers = generateAgents(self.gameData,self.buyerStrategies,self.sellerStrategies,self.disclosure)
        self.log = Log(self.gameData, self.buyerStrategies, self.sellerStrategies, self.disclosure)

    def simulate(self):
        for rnd in range(self.numRounds):
            roundData = roundSetup(*self.gameData)
            redemptionValues, tokenCosts, demand, supply, prices, peq, qeq  = roundData[0:7]
            [buyerReservationPrices, sellerReservationPrices, buyerSurplus, sellerSurplus, totalSurplus, buyerSurplusFrac, sellerSurplusFrac] = roundData[7:]
            self.log.addRound([rnd] + roundData)
            resetRounds(self.buyers, self.sellers, redemptionValues, tokenCosts)
            for period in range(self.numPeriods):
                resetPeriods(self.buyers, self.sellers)
                for step in range(self.numSteps):
                    resetSteps(self.buyers, self.sellers)
                    bids, asks = collectOffers(self.buyers, self.sellers)
                    currentAsk, currentAskIdx, currentBid, currentBidIdx = bestOffers(bids, asks)
                    price, buy, sell = trade(self.buyers, self.sellers, currentAsk, currentAskIdx, currentBid, currentBidIdx)
                    bprofit, sprofit = 0, 0
                    if price > 0:
                        self.buyers[currentBidIdx].transact(price)
                        self.sellers[currentAskIdx].transact(price)
                        bprofit = self.buyers[currentBidIdx].stepProfits
                        sprofit = self.sellers[currentAskIdx].stepProfits
                    self.log.addStep([rnd, period, step, bids, asks, currentBid, currentBidIdx, currentAsk, currentAskIdx, buy, sell, price, price>0, bprofit, sprofit])
                    observe(self.buyers, self.sellers, self.log.disclose())

    def graphSales(self, rnd, period):
        fig, ax = graphMarket(*self.gameData, *self.log.getRoundList(rnd))
        periodBids, periodAsks, periodPrices, periodSales = self.log.getPeriodActivity(rnd,period)
        plt.plot(range(1,len(periodPrices)+1), periodPrices, color='lightgreen', linestyle='--', label='Actual Prices')
        for i in range(numSteps):
            if (periodPrices[i] > 0):
                ax.scatter([periodSales[i]] * len(periodBids[i]), periodBids[i], s=10, alpha=0.5, c='purple')
                ax.scatter([periodSales[i]] * len(periodAsks[i]), periodAsks[i], s=10, alpha=0.5, c='darkgreen')
        plt.show()

    def graphOffers(self, rnd, period):
        periodBids, periodAsks, periodPrices, periodSales = self.log.getPeriodActivity(rnd,period)
        fig, ax = plt.subplots()
        ax.plot(periodBids, c='purple', linestyle='--', label='Bids')
        ax.plot(periodAsks, c='darkgreen', linestyle='--', label='Asks')
        ax.scatter(range(numSteps), periodPrices, c='black', label='Prices')
        ax.set_title('Bids (red), Asks (blue), and Prices (green) over trading steps')
        plt.show()
        
########################################################################
#                                Graphing
########################################################################

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Custom Settings
colors = sns.color_palette("husl", 5)
plot_settings = {
    'figure.figsize': (10, 7),
    'lines.linestyle': '--',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.grid': True,  # Add gridlines
    'font.family': 'serif',  # Use a serif font
    'font.size': 12,
}
plt.rcParams.update(plot_settings)
plt.style.use('seaborn-v0_8-darkgrid')
plt.ioff()

def graphMarket(gameTypes, numBuyers, numSellers, numTokens, numRounds, numPeriods, numSteps, redemptionValues, tokenCosts, 
                demand, supply, prices, peq, qeq, buyerReservationPrices, sellerReservationPrices, buyerSurplus,
                sellerSurplus, totalSurplus, buyerSurplusFrac, sellerSurplusFrac):
    fig, ax = plt.subplots() 
    ax.plot(demand, prices, label='Demand Curve', color=colors[4], alpha=1.0) 
    ax.plot(supply, prices, label='Supply Curve', color=colors[3], alpha=1.0) 
    ax.scatter(np.unique(demand)[::-1], buyerReservationPrices, label='Buyer Reservation Prices', marker='o', color=colors[4], alpha=0.5) 
    ax.scatter(np.unique(supply), sellerReservationPrices, label='Seller Reservation Prices', marker='o', color=colors[3], alpha=0.5) 
    ax.axhline(peq, color=colors[4], alpha = 0.5, label='Eqbm Prices', linestyle='-.')
    ax.axvline(qeq, color=colors[3], alpha = 0.5, label='Eqbm Quantities', linestyle='-.')

    # shade region
    demand_mask = (demand <= qeq) 
    supply_mask = (supply <= qeq)
    ax.fill_between(demand[demand_mask], peq*np.ones(len(demand[demand_mask])),
                    prices[demand_mask], color=colors[4], alpha=0.2, label='Buyer Surplus')
    ax.fill_between(supply[supply_mask], peq*np.ones(len(demand[supply_mask])),
                    prices[supply_mask], color=colors[3], alpha=0.2, label='Seller Surplus')
    ax.text(qeq * 0.1, peq * 1.2, f'{int(buyerSurplusFrac*100)}%', fontsize=10, color='black')
    ax.text(qeq * 0.1, peq * 0.8, f'{int(sellerSurplusFrac*100)}%', fontsize=10, color='black')
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Price')
    
    # Move the legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position('bottom')
    ax.set_yticks(np.arange(min(prices), max(prices) + 1, 5))
    ax.set_xticks(np.arange(0, max(max(demand), max(supply)) + 1, 1))

    # Information box with outline
    info_text = f'Game Type={gameTypes}\nNumber of Buyers = {numBuyers}\nNumber of Sellers = {numSellers}\nNumber of Tokens = {numTokens}'
    info_text2 = f'\nEquilibrium Price = {peq}\nEquilibrium Quantity = {qeq}'
    info_text3 = f'\nTotal Surplus = {totalSurplus}\nBuyer Surplus = {buyerSurplus}\nSeller Surplus = {sellerSurplus}'
    ax.text(qeq*0.75, 75, info_text + info_text2 + info_text3, alpha=0.7,
            fontsize=9, color='black', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    ax.set_title('Market Equilibrium', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig, ax

########################################################################
#                                Reinforcer
########################################################################
