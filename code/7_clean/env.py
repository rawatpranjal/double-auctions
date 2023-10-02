from setup import *
from agents import *

class BaseEnv:
    def __init__(self, metaData):
        print(metaData)
        self.gameData = metaData[0:8]
        self.gameTypes, self.numBuyers, self.numSellers, self.numTokens, self.numRounds, self.numPeriods, self.numSteps, self.seed = self.gameData
        self.disclosure, self.buyerStrategies, self.sellerStrategies = metaData[8:]
        self.buyers, self.sellers = generateAgents(self.gameData,self.buyerStrategies,self.sellerStrategies,self.disclosure)
        self.log = Log(self.gameData, self.buyerStrategies, self.sellerStrategies, self.disclosure)

    def simulate(self):
        for rnd in range(self.numRounds):
            roundData = roundSetup(*self.gameData)
            buyerValues, sellerCosts, demand, supply, prices, peq, qeq  = roundData[0:7]
            [buyerReservationPrices, sellerReservationPrices, buyerSurplus, sellerSurplus, totalSurplus, buyerSurplusFrac, sellerSurplusFrac] = roundData[7:]
            self.log.addRound([rnd] + roundData)
            resetRounds(self.buyers, self.sellers, buyerValues, sellerCosts)
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
        plt.plot(range(1,len(periodPrices)+1), periodPrices, color='darkgreen', linestyle='-.', label='Actual Prices')
        plt.plot([periodSales[i] for i in range(self.numSteps) if periodPrices[i] > 0],[periodBids[i][0] for i in range(self.numSteps) if periodPrices[i] > 0], color='red', linestyle='-.', label='Reinforcer Bids')
        for i in range(self.numSteps):
            if (periodPrices[i] > 0):
                ax.scatter(np.array([periodSales[i]] * len(periodBids[i][1:])), np.array(periodBids[i][1:]), s=10, alpha=0.5, c='purple')
                ax.scatter([periodSales[i]] * len(periodAsks[i]), periodAsks[i], s=10, alpha=0.5, c='blue')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        plt.show()


    def graphOffers(self, rnd, period):
        periodBids, periodAsks, periodPrices, periodSales = self.log.getPeriodActivity(rnd,period)
        fig, ax = plt.subplots()
        colors = customGraphSettings()
        ax.plot(periodBids, c='purple', linestyle='--', label='Bids')
        ax.plot(periodAsks, c='darkgreen', linestyle='--', label='Asks')
        ax.scatter(range(self.numSteps), periodPrices, c='black', label='Prices')
        ax.set_title('Offers over time')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)
        plt.show()

import gymnasium as gym
from gymnasium import spaces

class SingleAgentEnv(gym.Env):
    def __init__(self, metaData):
        self.gameData = metaData[0:8]
        self.gameTypes, self.numBuyers, self.numSellers, self.numTokens, self.numRounds, self.numPeriods, self.numSteps, self.seed = self.gameData
        self.disclosure, self.buyerStrategies, self.sellerStrategies = metaData[8:]
        self.buyers, self.sellers = generateAgents(self.gameData,self.buyerStrategies,self.sellerStrategies,self.disclosure)
        self.log = Log(self.gameData, self.buyerStrategies, self.sellerStrategies, self.disclosure)
        self.roundData = roundSetup(*self.gameData)
        self.buyerValues, self.sellerCosts, self.demand, self.supply, self.prices, self.peq, self.qeq  = self.roundData[0:7]
        [self.buyerReservationPrices, self.sellerReservationPrices, self.buyerSurplus, self.sellerSurplus, self.totalSurplus, self.buyerSurplusFrac, self.sellerSurplusFrac] = self.roundData[7:]
        self.log.addRound([0] + self.roundData)
        resetRounds(self.buyers, self.sellers, self.buyerValues, self.sellerCosts)
        self.depth = 9
        self.numStates = 8 + 6 * self.depth
        self.action_space = spaces.Box(-1,1,(1,),dtype=np.float32)
        self.observation_space = spaces.Box(-1000,1000,(self.numStates,),dtype=np.float32)
        self.timePeriod = 0
        self.timeStep = 0

    def reset(self, seed = None):
        resetPeriods(self.buyers, self.sellers)
        resetSteps(self.buyers, self.sellers)
        agent = self.buyers[0]
        agentState = [self.timeStep, agent.stepTokenValue, agent.stepBid, agent.stepAsk, agent.stepTrades, agent.stepProfits, agent.periodTrades, agent.periodProfits]
        history = [-1, -1, -1, -1, -1, -1] * self.depth
        state = np.nan_to_num(np.array(agentState + history, dtype = np.float32), nan=-9)
        self.numStates = state.shape[0]
        return state, {}

    def step(self, action):
        # convert action into bid
        minBid = 0
        maxBid = 100
        frac = (action.item()+1)/2
        bids, asks = collectOffers(self.buyers, self.sellers)
        bids[0] = np.round(frac * minBid + (1-frac) * maxBid,1)

        # simulate market
        currentAsk, currentAskIdx, currentBid, currentBidIdx = bestOffers(bids, asks)
        price, buy, sell = trade(self.buyers, self.sellers, currentAsk, currentAskIdx, currentBid, currentBidIdx)
        bprofit, sprofit = 0, 0
        if price > 0:
            self.buyers[currentBidIdx].transact(price)
            self.sellers[currentAskIdx].transact(price)
            bprofit = self.buyers[currentBidIdx].stepProfits
            sprofit = self.sellers[currentAskIdx].stepProfits
        self.log.addStep([0, self.timePeriod, self.timeStep, bids, asks, currentBid, currentBidIdx, currentAsk, currentAskIdx, buy, sell, price, price>0, bprofit, sprofit])
        observe(self.buyers, self.sellers, self.log.disclose())

        # compute reward and transition
        reward = 0.0
        if price > 0 and currentBidIdx == 0:
            reward = np.nan_to_num(bprofit,nan=0)
        agent = self.buyers[0]
        agentState = [self.timeStep, agent.stepTokenValue, agent.stepBid, agent.stepAsk, agent.stepTrades, agent.stepProfits, agent.periodTrades, agent.periodProfits]
        history = self.buyers[0].df.iloc[-self.depth:][['currentBid', 'currentAsk', 'buy', 'sell', 'price', 'price']].values.reshape(-1,).tolist()
        if len(history) == (6 * self.depth):
            pass
        else:
            history = [-1] * (6 * self.depth)
        newState = np.nan_to_num(np.array(agentState + history, dtype = np.float32), nan=-9)
        # check termination
        if self.timeStep == self.numSteps - 1:
            terminated = True
            self.timePeriod += 1
            self.timeStep = 0
        else:
            self.timeStep += 1
            terminated = False
        infos = {"TimeLimit.truncated":True}
        truncated = False
        resetSteps(self.buyers, self.sellers)
        return newState, reward, terminated, truncated, infos

    def graphSales(self, rnd, period):
        fig, ax = graphMarket(*self.gameData, *self.log.getRoundList(rnd))
        periodBids, periodAsks, periodPrices, periodSales = self.log.getPeriodActivity(rnd,period)
        plt.plot(range(1,len(periodPrices)+1), periodPrices, color='darkgreen', linestyle='--', label='Actual Prices')
        print(np.array([periodBids[i][0] for i in range(self.numSteps) if periodPrices[i] > 0]))
        plt.plot([periodSales[i] for i in range(self.numSteps) if periodPrices[i] > 0],[periodBids[i][0] for i in range(self.numSteps) if periodPrices[i] > 0], color='darkorange', linestyle='--', label='Reinforcer Bids')
        for i in range(self.numSteps):
            if (periodPrices[i] > 0):
                ax.scatter(np.array([periodSales[i]] * len(periodBids[i][1:])), np.array(periodBids[i][1:]), s=10, alpha=0.5, c='purple')
                ax.scatter([periodSales[i]] * len(periodAsks[i]), periodAsks[i], s=10, alpha=0.5, c='blue')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        plt.show()

    def graphOffers(self, rnd, period):
        periodBids, periodAsks, periodPrices, periodSales = self.log.getPeriodActivity(rnd,period)
        fig, ax = plt.subplots()
        ax.plot(np.array(periodBids)[:, 0], c='red', linestyle='--', label='Bids')
        ax.plot(np.array(periodBids)[:, 1:], c='purple', linestyle='--', label='Bids')
        ax.plot(periodAsks, c='darkgreen', linestyle='--', label='Asks')
        print(len(range(self.numSteps)))
        print(len(periodPrices))
        ax.scatter(range(self.numSteps), periodPrices, c='black', label='Prices')
        ax.set_title('Bids (red), Asks (blue), and Prices (green) over trading steps')
        plt.show()