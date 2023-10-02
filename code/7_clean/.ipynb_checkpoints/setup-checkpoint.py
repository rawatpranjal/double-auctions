####################################################################
#                          GAME SETUP
####################################################################

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
np.printoptions(precision=2, suppress=True)

def gameTypeToUpperBounds(gameType='5555'):
    UB1 = int(gameType[0])
    UB2 = int(gameType[1])
    UB3 = int(gameType[2])
    UB4 = int(gameType[3])
    return UB1, UB2, UB3, UB4
    
def genTokenValues(UB1 = 1, UB2 = 0, UB3 = 0, UB4 = 1, numBuyers=4, numSellers=4, numTokens=4, seed = None):
    if seed != None:
        np.random.seed(seed)
    baseValues = np.random.uniform(0, UB1, (numBuyers + numSellers, numTokens)) 
    agentDifferentiator = np.random.uniform(0, UB2, (numBuyers + numSellers, 1)) 
    tokenDifferentiator = np.random.uniform(0, UB3, (1, numTokens))  
    buyerDifferentiator = np.random.uniform(0, UB4, (numBuyers, 1)) 
    tokenValues = baseValues + agentDifferentiator + tokenDifferentiator + np.r_[buyerDifferentiator, np.zeros((numSellers, 1))]
    normTokenValues = np.round(((tokenValues - tokenValues.min()) / (tokenValues.max() - tokenValues.min())) * 100,1)
    return normTokenValues

def sortTokens(tokenValues, numBuyers, numSellers):
    buyerValues = tokenValues[0:numBuyers,:]
    descIdx = np.argsort(buyerValues, axis=1)[:,::-1] # buyer values fall
    buyerValues = np.take_along_axis(buyerValues, descIdx, axis=1)
    
    sellerCosts = tokenValues[numBuyers:(numBuyers + numSellers), :]
    ascIdx = np.argsort(sellerCosts, axis = 1) # seller costs rise
    sellerCosts = np.take_along_axis(sellerCosts, ascIdx, axis=1)
    return buyerValues, sellerCosts

def demandSupplySchedules(buyerValues, sellerCosts, granularity = 200, minprice = 0, maxprice = 100):
    prices = np.linspace(minprice, maxprice, granularity)
    demand = np.zeros((granularity), dtype='int')
    supply = np.zeros((granularity), dtype='int')
    for i, price in enumerate(prices):
        demand[i] = np.round(np.sum(price < buyerValues))
        supply[i] = np.round(np.sum(price > sellerCosts))
    return demand, supply, prices

def equilibrium(demand,supply,prices):
    clearingPrices, clearingQuantity = [], np.nan
    for i, price in enumerate(prices):
        if demand[i] == supply[i]:
            clearingPrices.append(price)
            clearingQuantity = demand[i]
            
    if len(clearingPrices) == 0: # if no clearing price
        minDiffIdx = np.argmin(np.abs(demand-supply))
        clearingPrices = prices[minDiffIdx] 
        clearingQuantity = demand[minDiffIdx]

    clearingPrice = np.nanmean(clearingPrices)
    clearingPrice = np.round(clearingPrice,1)
    return np.round(clearingPrice,1), clearingQuantity 

def reservationPrices(demand, supply,prices):
    demandChangeIdx = np.where(demand[:-1] != demand[1:], 1, 0)
    demandChangeIdx = np.pad(demandChangeIdx, (0,1))
    buyerReservationPrices = prices[demandChangeIdx==1] # price at which one more token would get sold

    supplyChangeIdx = np.where(supply[:-1] != supply[1:], 1, 0)
    supplyChangeIdx = np.pad(supplyChangeIdx, (0,1))
    sellerReservationPrices = prices[supplyChangeIdx==1]
    return buyerReservationPrices, sellerReservationPrices

def surplus(buyerValues, sellerCosts, clearingPrice, clearingQuantity):
    buyerSurplus = np.round(np.sum(buyerValues - clearingPrice, where = buyerValues - clearingPrice > 0),1)
    sellerSurplus = np.round(np.sum(clearingPrice - sellerCosts, where = clearingPrice - sellerCosts > 0),1)
    totalSurplus = np.round(buyerSurplus + sellerSurplus)
    buyerSurplusFrac = np.round(buyerSurplus/totalSurplus,1)
    sellerSurplusFrac = np.round(sellerSurplus/totalSurplus,1)
    return buyerSurplus, sellerSurplus, totalSurplus, buyerSurplusFrac, sellerSurplusFrac

def roundSetup(gameTypes, numBuyers, numSellers, numTokens, numRounds, numPeriods, numSteps, seed):
    UB1, UB2, UB3, UB4 = gameTypeToUpperBounds(gameType='5555')
    tokenValues = genTokenValues(UB1, UB2, UB3, UB4, numBuyers, numSellers, numTokens, seed = seed)
    buyerValues, sellerCosts = sortTokens(tokenValues, numBuyers, numSellers)
    demand, supply, prices = demandSupplySchedules(buyerValues,sellerCosts)
    clearingPrice, clearingQuantity = equilibrium(demand,supply,prices)
    buyerReservationPrices, sellerReservationPrices = reservationPrices(demand, supply,prices)
    buyerSurplus, sellerSurplus, totalSurplus, buyerSurplusFrac, sellerSurplusFrac = surplus(buyerValues,sellerCosts,
                                                                                             clearingPrice, clearingQuantity)
    metadata = []
    metadata += [buyerValues, sellerCosts, demand, supply, prices, clearingPrice, clearingQuantity]
    metadata += [buyerReservationPrices, sellerReservationPrices, buyerSurplus, sellerSurplus, totalSurplus, buyerSurplusFrac, sellerSurplusFrac]
    return metadata

####################################################################
#                          AGENT HELPER
####################################################################

def resetRounds(buyers, sellers, buyerValues, sellerCosts):
    for i, buyer in enumerate(buyers):
        buyer.resetRound(buyerValues[i,:])
    for i, seller in enumerate(sellers):
        seller.resetRound(sellerCosts[i,:])

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
        return price - value
    else:
        return value - price

def updatePolicy(buyers, sellers):
    for buyer in buyers:
        if buyer.reinforcer == 1:
            buyer.train()
    for seller in sellers:
        if seller.reinforcer == 1:
            seller.train()

def updateStates(buyers, sellers):
    for buyer in buyers:
        if buyer.reinforcer == 1:
            buyer.observe()
    for seller in sellers:
        if seller.reinforcer == 1:
            seller.observe()


####################################################################
#                          GRAPH
####################################################################

def customGraphSettings():
    colors = sns.color_palette("husl", 5)
    plot_settings = {
        'figure.figsize': (10, 7),
        'lines.linestyle': '--',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.grid': True,
        'font.family': 'serif',
        'font.size': 12}
    plt.rcParams.update(plot_settings)
    plt.style.use('seaborn-v0_8-darkgrid')
    return colors

def graphMarket(gameTypes, numBuyers, numSellers, numTokens, numRounds, numPeriods, numSteps, seed,
                buyerValues, sellerCosts, 
                demand, supply, prices, clearingPrice, clearingQuantity,
                buyerReservationPrices, sellerReservationPrices,
                buyerSurplus, sellerSurplus, totalSurplus, buyerSurplusFrac, sellerSurplusFrac):

    # plot schedules
    colors = customGraphSettings()
    fig, ax = plt.subplots() 
    ax.plot(demand, prices, label='Demand Curve', color=colors[4], alpha=1.0) 
    ax.plot(supply, prices, label='Supply Curve', color=colors[3], alpha=1.0) 
    ax.scatter(np.unique(demand)[::-1][1:], buyerReservationPrices,
               label='Buyer Reservation Prices', marker='o', color=colors[4], alpha=0.5)
    ax.scatter(np.unique(supply)[1:], sellerReservationPrices,
               label='Seller Reservation Prices', marker='o', color=colors[3], alpha=0.5)
    ax.axhline(clearingPrice, color=colors[4], alpha = 0.5, label='Eqbm Prices', linestyle='-.')
    ax.axvline(clearingQuantity, color=colors[3], alpha = 0.5, label='Eqbm Quantities', linestyle='-.')

    # shade
    demand_mask = (demand <= clearingQuantity) 
    supply_mask = (supply <= clearingQuantity)
    ax.fill_between(demand[demand_mask], clearingPrice*np.ones(len(demand[demand_mask])),
                    prices[demand_mask], color=colors[4], alpha=0.2, label='Buyer Surplus')
    ax.fill_between(supply[supply_mask], clearingPrice*np.ones(len(demand[supply_mask])),
                    prices[supply_mask], color=colors[3], alpha=0.2, label='Seller Surplus')
    ax.text(clearingQuantity * 0.1, clearingPrice * 1.2, f'{int(buyerSurplusFrac*100)}%', fontsize=10, color='black')
    ax.text(clearingQuantity * 0.1, clearingPrice * 0.8, f'{int(sellerSurplusFrac*100)}%', fontsize=10, color='black')
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Price')
    
    # legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position('bottom')
    ax.set_yticks(np.arange(min(prices), max(prices) + 1, 5))
    ax.set_xticks(np.arange(0, max(max(demand), max(supply)) + 1, 1))

    # infobox
    text1 = f'Game Type={gameTypes} \nNumber of Buyers = {numBuyers} \nNumber of Sellers = {numSellers} \nNumber of Tokens = {numTokens}'
    text2 = f'\nEquilibrium Price = {clearingPrice} \nEquilibrium Quantity = {clearingQuantity}'
    text3 = f'\nTotal Surplus = {totalSurplus} \nBuyer Surplus = {buyerSurplus} \nSeller Surplus = {sellerSurplus}'
    ax.text(clearingQuantity*0.75, 75, text1 + text2 + text3, alpha=0.7,
            fontsize=9, color='black', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    ax.set_title('Market Equilibrium', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig, ax

####################################################################
#                          LOG
####################################################################

class Log:
    def __init__(self, gameData, buyerStrategies, sellerStrategies,disclosure):
        self.gameTypes, self.numBuyers, self.numSellers, self.numTokens, self.numRounds, self.numPeriods, self.numSteps, self.seed = gameData
        self.stepData = pd.DataFrame(columns=['rnd', 'period', 'step', 'bids','asks','currentBid','currentBidIdx','currentAsk',
                                             'currentAskIdx','buy','sell','price','sale', 'bprofit', 'sprofit'])
        self.roundData = pd.DataFrame(columns=['rnd', 'buyerValues', 'sellerCosts', 'demand', 'supply', 'prices', 'peq', 'qeq', 
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
