
class Trader:
    def __init__(self, gameData, index, buyer):
        self.gameType, self.numBuyers, self.numSellers, self.numTokens, self.numRounds, self.numPeriods, self.numSteps, self.seed = gameData
        self.index, self.buyer = index, buyer

    def startRound(self, tokenValues):
        self.roundTokens = tokenValues

    def startPeriod(self):
        self.periodTokens = self.roundTokens
    
    def startStep(self):
        self.stepTokenValue = np.nan
        if len(self.periodTokens)!=0:
            self.stepTokenValue = self.periodTokens[0]

class ZIC(Trader):
    def __init__(self, gameData, index, buyer):
        super().__init__(gameData, index, buyer)
    
    def bid(self):
        self.stepBid = np.nan
        if self.stepTokenValue>=0:
            self.stepBid = np.random.uniform(self.stepTokenValue*0.5,self.stepTokenValue*1.0, 1).item()
        return np.round(self.stepBid,1)
        
    def ask(self):
        self.stepAsk = np.nan
        if self.stepTokenValue>=0:
            self.stepAsk = np.random.uniform(self.stepTokenValue*1.0,self.stepTokenValue*1.5, 1).item()
        return np.round(self.stepAsk, 1)
