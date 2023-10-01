########################################################################
#                               LOG
########################################################################

from setup import *
from agents import *

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
#                               BASE ENV
########################################################################

class BaseEnv:
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
        for i in range(self.numSteps):
            if (periodPrices[i] > 0):
                ax.scatter([periodSales[i]] * len(periodBids[i]), periodBids[i], s=10, alpha=0.5, c='purple')
                ax.scatter([periodSales[i]] * len(periodAsks[i]), periodAsks[i], s=10, alpha=0.5, c='darkgreen')
        plt.show()

    def graphOffers(self, rnd, period):
        periodBids, periodAsks, periodPrices, periodSales = self.log.getPeriodActivity(rnd,period)
        fig, ax = plt.subplots()
        ax.plot(periodBids, c='purple', linestyle='--', label='Bids')
        ax.plot(periodAsks, c='darkgreen', linestyle='--', label='Asks')
        ax.scatter(range(self.numSteps), periodPrices, c='black', label='Prices')
        ax.set_title('Bids (red), Asks (blue), and Prices (green) over trading steps')
        plt.show()