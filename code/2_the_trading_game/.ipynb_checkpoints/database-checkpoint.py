from packages import *
from setup import plot_period_results

########################################################################
######################## Database
########################################################################

class Database:
    def __init__(self, game_metadata, buyers, sellers):
        # input public information, buyers and sellers
        self.game_metadata = game_metadata
        self.nrounds, self.nperiods, self.ntokens, self.nbuyers, self.nsellers, self.R1, self.R2, self.R3, self.R4, self.minprice, self.maxprice, self.nsteps = game_metadata       
        self.buyers, self.sellers = buyers, sellers
        self.step_data = pd.DataFrame(columns=['rnd', 'period', 'step', 'bids','asks','current_bid','current_bid_idx','current_ask','current_ask_idx','buy','sell','price','sale', 'bprofit', 'sprofit', 'p_eqbm','q_eqbm'])
        self.round_data = pd.DataFrame(columns=['rnd', 'token_values', 'p_eqbm', 'q_eqbm'])
        
    def add_step(self, data):
        self.step_data.loc[len(self.step_data.index)] = data

    def add_round(self, data):        
        self.round_data.loc[len(self.round_data.index)] = data
        
    def get_period(self, rnd, period):
        temp = self.step_data[(self.step_data.rnd==rnd) & (self.step_data.period==period)]
        temp['sales'] =  temp['sale'].cumsum()
        temp = temp[['step', 'bids', 'asks', 'current_bid', 'current_bid_idx', 'current_ask','current_ask_idx','price', 'bprofit', 'sprofit', 'sales']]
        return temp

    def get_round(self, rnd):
        print(f"Rounds:{self.nrounds}, Periods{self.nperiods}, Tokens:{self.ntokens}, Buyers:{self.nbuyers}, Sellers{self.nsellers}, R1 to R4:{self.R1, self.R2, self.R3, self.R4}")
        temp = self.round_data[(self.round_data.rnd==rnd)]
        temp = temp[['token_values', 'p_eqbm', 'q_eqbm']]
        return temp

    def get_token_values(self, rnd):
        return self.round_data[(self.round_data.rnd==rnd)].token_values
    
    def graph_period(self, rnd, period):
        period_bids = list(self.get_period(rnd, period).bids)
        period_asks = list(self.get_period(rnd, period).asks)
        period_prices = list(self.get_period(rnd, period).price)
        period_sales = np.cumsum(np.where(self.get_period(rnd, period).price > 0,1,0))
        token_values = self.get_round(rnd).token_values.item()
        ntokens, nbuyers, nsellers, nsteps, R1, R2, R3, R4 = self.ntokens, self.nbuyers, self.nsellers, self.nsteps, self.R1, self.R2, self.R3, self.R4
        plot_period_results(period_bids, period_asks, period_prices, period_sales, token_values, ntokens, nbuyers, nsellers, nsteps, R1, R2, R3, R4)

