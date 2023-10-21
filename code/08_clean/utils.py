####################################################################
#                          GAME SETUP
####################################################################

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import warnings 
warnings.filterwarnings('ignore')
import torch as th
th.autograd.set_detect_anomaly(True)
np.printoptions(precision=2, suppress=True)

def gameTypeToUpperBounds(gameType='1111'):
    UB1 = int(gameType[0])
    UB2 = int(gameType[1])
    UB3 = int(gameType[2])
    UB4 = int(gameType[3])
    return UB1, UB2, UB3, UB4
    
def genTokenValues(UB1 = 1, UB2 = 1, UB3 = 1, UB4 = 1, numBuyers=4, numSellers=4, numTokens=4, seed = None):
    if seed != None:
        np.random.seed(seed)
    if UB1 == 1:
        baseValues = np.random.normal(50, 20, (numBuyers + numSellers, numTokens)) 
    if UB2 == 1:
        agentDifferentiator = np.random.normal(0, 1, (numBuyers + numSellers, 1))
    if UB3 == 1:
        tokenDifferentiator = np.random.normal(0, 1, (1, numTokens)) 
    if UB4 == 1:
        buyerDifferentiator = np.random.normal(20, 1, (numBuyers, 1)) 
    tokenValues = baseValues + agentDifferentiator + tokenDifferentiator + np.r_[buyerDifferentiator, np.zeros((numSellers, 1))]
    #normTokenValues = np.round(((tokenValues - tokenValues.min()) / (tokenValues.max() - tokenValues.min())) * 100,1)
    return np.clip(tokenValues,0,100)

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
    UB1, UB2, UB3, UB4 = gameTypeToUpperBounds(gameType='1111')
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

def resetTokens(buyers, sellers, buyerValues, sellerCosts):
    for i, buyer in enumerate(buyers):
        buyer.startRound(buyerValues[i,:])
    for i, seller in enumerate(sellers):
        seller.startRound(sellerCosts[i,:])

def startRounds(gameData, log, buyers, sellers, rnd):
    roundData = roundSetup(*gameData)
    buyerValues, sellerCosts = roundData[0:2]
    log.addRound([rnd] + roundData)
    resetTokens(buyers, sellers, buyerValues, sellerCosts)

def endRounds(buyers, sellers):
    for i,agent in enumerate(buyers + sellers):
        agent.endRound()

def startPeriods(buyers, sellers):
    for i,agent in enumerate(buyers + sellers):
        agent.startPeriod()

def endPeriods(buyers, sellers):
    for i,agent in enumerate(buyers + sellers):
        agent.endPeriod()

def startSteps(buyers, sellers):
    for i,agent in enumerate(buyers + sellers):
        agent.startStep()

def endSteps(buyers, sellers):
    for i,agent in enumerate(buyers + sellers):
        agent.endStep()

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

def trade(buyers, sellers, currentAsk, currentAskIdx, currentBid, currentBidIdx, pricefrac = 0.5):
        price, buy, sell = np.nan, np.nan, np.nan
        if (currentBidIdx >= 0) and (currentAskIdx >= 0):
            buy = buyers[currentBidIdx].buy(currentBid,currentAsk)
            sell = sellers[currentAskIdx].sell(currentBid,currentAsk)
            if buy and not sell:
                price = currentAsk
            elif sell and not buy:
                price = currentBid
            elif sell and buy: 
                price = pricefrac*currentBid + (1-pricefrac)*currentAsk
        return price, buy, sell

def observe(buyers, sellers, data):
    for i, agent in enumerate(buyers+sellers):
        agent.df = pd.concat([agent.df, data.to_frame().T], ignore_index=True)

def profit(value,price,buyer):
    if buyer == 0:
        return price - value
    else:
        return value - price

def agentsObserve(buyers, sellers):
    for buyer in buyers:
        #if buyer.reinforcer == 1:
        #    buyer.observe()
        pass
    for seller in sellers:
        #if seller.reinforcer == 1:
        #    seller.observe()
        pass

def agentsTrain(buyers, sellers):
    for buyer in buyers:
        if buyer.reinforcer == 1:
            buyer.train()
    for seller in sellers:
        if seller.reinforcer == 1:
            seller.train()

def generateState(agent):
    counters = [agent.periodSteps, agent.periodTrades] #, agent.periodProfits, agent.stepTokenValue
    disclosureLength = len(agent.disclosure)
    if (disclosureLength == 0) | (agent.depth == 0):
        activityLog = []
    else:   
        if agent.gameSteps >= agent.depth:
            agent.disclosureCopy = deepcopy(agent.disclosure)
            bidsDisclose, asksDisclose = False, False
            if 'bids' in agent.disclosure:
                agent.disclosureCopy.remove('bids')
                bidsDisclose = True
            if 'asks' in agent.disclosure:
                agent.disclosureCopy.remove('asks')
                asksDisclose = True
            
            activityLog = [[]]
            for i in range(1, agent.depth+1):
                activityLog[0] += agent.df.iloc[-i][agent.disclosureCopy].tolist()           
                if bidsDisclose:
                    activityLog[0] += agent.df.iloc[-i].bids
                if asksDisclose:
                    activityLog[0] += agent.df.iloc[-i].asks
            activityLog = activityLog[0]
        else:
            bidsDisclose, asksDisclose = False, False
            if 'bids' in agent.disclosure:
                disclosureLength -= 1
                bidsDisclose = True
            if 'asks' in agent.disclosure:
                disclosureLength -= 1
                asksDisclose = True
            activityLog = [-1] * (disclosureLength*agent.depth + 
                                  bidsDisclose*agent.depth*agent.numBuyers+asksDisclose*agent.depth*agent.numSellers)
        
    state = counters + activityLog
    cleanState = [-1 if np.isnan(x) else x for x in state]
    return cleanState


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
    ax.plot(demand, prices, label='Demand Curve', color=colors[4], alpha=1.0, linestyle='dashed') 
    ax.plot(supply, prices, label='Supply Curve', color=colors[3], alpha=1.0, linestyle='dashed') 
    #ax.scatter(np.unique(demand)[::-1][1:], buyerReservationPrices,
               #label='Buyer Reservation Prices', marker='o', color=colors[4], alpha=0.5)
    #ax.scatter(np.unique(supply)[1:], sellerReservationPrices,
               #label='Seller Reservation Prices', marker='o', color=colors[3], alpha=0.5)
    #ax.axhline(clearingPrice, color=colors[4], alpha = 0.5, label='Eqbm Prices', linestyle='dashed')
    #ax.axvline(clearingQuantity, color=colors[3], alpha = 0.5, label='Eqbm Quantities', linestyle='solid')

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
    ax.set_title('Dynamic Double Auction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig, ax

####################################################################
#                          LOG
####################################################################

class Log:
    def __init__(self, gameData,disclosure):
        self.gameData = gameData
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
        return np.array(periodBids), np.array(periodAsks), np.array(periodPrices), np.array(periodSales)

    def getRound(self, rnd):
        return self.roundData[(self.roundData.rnd==rnd)]

    def getRoundList(self, rnd):
        return self.getRound(rnd).values.tolist()[0][1:]


    def graphSales(self, rnd, period, trackBuyersIdx = [], trackSellersIdx = []):
        colors = customGraphSettings()
        fig, ax = graphMarket(*self.gameData, *self.getRoundList(rnd))
        periodBids, periodAsks, periodPrices, periodSales = self.getPeriodActivity(rnd,period)
        plt.plot(range(1,len(periodPrices)+1), periodPrices, color='black', linestyle='dashdot', label='Actual Prices')
        
        for j in range(self.numBuyers):
            y = [periodSales[i] for i in range(self.numSteps) if periodPrices[i] > 0]
            x = [periodBids[i][j] for i in range(self.numSteps) if periodPrices[i] > 0]
            if j in trackBuyersIdx:
                plt.plot(y, x, linestyle='dotted', color = 'red')
            else:
                plt.plot(y, x, linestyle='dotted', color = 'red', alpha = 0.2)
                
        for j in range(self.numSellers):
            y = [periodSales[i] for i in range(self.numSteps) if periodPrices[i] > 0]
            x = [periodAsks[i][j] for i in range(self.numSteps) if periodPrices[i] > 0]
            if j in trackSellersIdx:
                plt.plot(y, x, linestyle='dotted', color = 'blue')
            else:
                plt.plot(y, x, linestyle='dotted', color = 'blue', alpha = 0.2)
                
        for i in range(self.numSteps):
            if (periodPrices[i] > 0):
                ax.scatter([periodSales[i]] * len(periodBids[i]), periodBids[i], s=10, alpha=0.2, c='red')
                ax.scatter([periodSales[i]] * len(periodAsks[i]), periodAsks[i], s=10, alpha=0.2, c='blue')
                
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        ax.set_title('Dynamic Double Auction - Transactions', fontsize=16, fontweight='bold')
        plt.text(0.90, 0.10, f'Round: {rnd}', transform=plt.gca().transAxes, alpha=0.5)
        plt.text(0.90, 0.05, f'Period: {period}', transform=plt.gca().transAxes, alpha=0.5)
        plt.show()
    

    def graphOffers(self, rnd, period, trackBuyersIdx = [], trackSellersIdx = []):
        periodBids, periodAsks, periodPrices, periodSales = self.getPeriodActivity(rnd,period)
        fig, ax = plt.subplots()
        colors = customGraphSettings()
        notTrackBuyers = [i for i in range(self.numBuyers) if i not in trackBuyersIdx]
        notTrackSellers = [i for i in range(self.numBuyers) if i not in trackSellersIdx]
        ax.plot(periodBids[:,notTrackBuyers], c='red', linestyle='dotted', alpha = 0.3)
        ax.plot(periodBids[:,trackBuyersIdx], c='red', linestyle='dotted', alpha = 1.0)
        ax.scatter(range(self.numSteps), np.max(periodBids, axis = 1), c='red', label='Winning Bids', alpha = 0.5)
        ax.plot(periodAsks[:,notTrackSellers], c='blue', linestyle='dotted', alpha = 0.3)
        ax.plot(periodAsks[:,trackSellersIdx], c='blue', linestyle='dotted', alpha = 1.0)
        ax.scatter(range(self.numSteps), np.min(periodAsks, axis = 1), c='blue', label='Winning Asks', alpha = 0.5)
        #ax.plot(periodBids, c='red', linestyle='dotted', label='Bids')
        #ax.plot(periodAsks, c='blue', linestyle='dotted', label='Asks')
        ax.plot(range(self.numSteps), periodPrices, c='black', label='Prices', linestyle='dashdot')
        #ax.scatter(range(self.numSteps), periodPrices, c='black', label='Prices')
        ax.set_title('Dynamic Double Auction - Offers', fontsize=16, fontweight='bold')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        plt.xlabel('Price')
        plt.ylabel('Time Step')
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        ax.set_yticks(np.arange(0, 100, 5))
        ax.set_xticks(np.arange(0, len(periodSales), 1))
        plt.text(0.90, 0.10, f'Round: {rnd}', transform=plt.gca().transAxes, alpha=0.5)
        plt.text(0.90, 0.05, f'Period: {period}', transform=plt.gca().transAxes, alpha=0.5)
        plt.show()


    def graphLearning(self, trackBuyersIdx = [], trackSellersIdx = [], rolling_window = 1, type = 'period'):
        df = self.stepData
        fig, ax = plt.subplots()
        colors = customGraphSettings()
        for j in range(self.numBuyers):
            rolling_mean = df[df.currentBidIdx == j][[type, 'bprofit']].groupby([type]).sum().rolling(rolling_window).mean()
            if j in trackBuyersIdx:
                ax.plot(rolling_mean, color = 'red', alpha = 1.0, linestyle = 'dotted', label = f'Bidder {j}')
            else:
                ax.plot(rolling_mean, color = 'red', alpha = 0.3, linestyle = 'dotted')
                
        for j in range(self.numSellers):
            rolling_mean = df[df.currentAskIdx == j][[type, 'sprofit']].groupby([type]).sum().rolling(rolling_window).mean()
            if j in trackSellersIdx:
                ax.plot(rolling_mean, color = 'blue', alpha = 1.0, linestyle = 'dotted', label = f'Asker {j}')
            else:
                ax.plot(rolling_mean, color = 'blue', alpha = 0.3, linestyle = 'dotted')
        
        ax.set_title('Learning Curves', fontsize=16, fontweight='bold')
        plt.xlabel('Period')
        plt.ylabel('Avg Profit')
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        #ax.set_yticks(np.arange(0, 100, 5))
        #ax.set_xticks(np.arange(0, 100, 20))
        #ax.text(0.80, 0.10, f'Round: {rnd}', transform=plt.gca().transAxes, alpha=0.5)
        #ax.text(0.80, 0.05, f'Period: {period}', transform=plt.gca().transAxes, alpha=0.5)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        plt.ylim(ymin=0)
        plt.show()

    def init(self, ax, totalProfit, period):
        ax.clear()
        ax.set_title('Dynamic Double Auction - Offers', fontsize=16, fontweight='bold')
        ax.set_xlabel('Price')
        ax.set_ylabel('Time Step')
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 100)
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        ax.set_yticks(np.arange(0, 100, 5))
        ax.set_xticks(np.arange(0, 10, 1))
        ax.text(0.70, 0.10, f'Period: {period}', transform=ax.transAxes, alpha=0.5)
        ax.text(0.70, 0.05, f'Profit: {np.round(totalProfit, 1)}', transform=ax.transAxes, alpha=0.5)

    def update(self, ax, period, periodBids, periodAsks, trackBuyersIdx, trackSellersIdx):
        df = self.getPeriod(0, period)    
        try:
            totalProfit = df[df.currentBidIdx == 0].groupby('currentBidIdx').sum().bprofit.item()
        except:
            totalProfit = 0
        self.init(ax, totalProfit, period)

        notTrackSellers = [i for i in range(self.numSellers) if i not in trackSellersIdx]
        notTrackBuyers = [i for i in range(self.numBuyers) if i not in trackBuyersIdx]
        ax.plot(np.array(periodBids)[:, notTrackBuyers], c='red', linestyle='dotted', alpha=0.3)
        ax.plot(np.array(periodBids)[:, trackBuyersIdx], c='red', linestyle='dotted', alpha=1.0)
        ax.plot(np.array(periodAsks)[:, notTrackSellers], c='blue', linestyle='dotted', alpha=0.3)
        ax.plot(np.array(periodAsks)[:, trackSellersIdx], c='blue', linestyle='dotted', alpha=1.0)

        a = np.max(np.array(periodBids), axis=0)
        b = np.max(np.array(periodBids)[:, trackBuyersIdx], axis=0)
        timesteps, maxbids = [], []
        for i in range(np.array(periodBids).shape[0]):
            if a[i] == b[i]:
                timesteps.append(i)
                maxbids.append(a[i])
        
        ax.scatter(timesteps, maxbids, c='green', marker='o', label='Winning Bid')   
        print(timesteps, maxbids)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Offers')
        ax.legend()

    def clean_frames_folder(self):
        for filename in os.listdir('frames'):
            file_path = os.path.join('frames', filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        if os.path.exists('frames'):
            os.rmdir('frames')

    def graphTraining(self, rnd=0, skip=1, maxNumPeriods=1000, trackBuyersIdx=[0], trackSellersIdx=[]):
        fig, ax = plt.subplots()

        if not os.path.exists('frames'):
            os.makedirs('frames')

        for period in range(1, maxNumPeriods, skip):
            periodBids, periodAsks, periodPrices, periodSales = self.getPeriodActivity(rnd, period)
            self.update(ax, period, periodBids, periodAsks, trackBuyersIdx, trackSellersIdx)
            plt.savefig(f'frames/period_{period}_frame.png')

        plt.close(fig)

        if os.path.exists('animation.gif'):
            os.remove('animation.gif')

        images = []
        for period in range(1, maxNumPeriods, skip):
            filename = f'frames/period_{period}_frame.png'
            images.append(imageio.imread(filename))
        imageio.mimsave('animation.gif', images)
        self.clean_frames_folder()