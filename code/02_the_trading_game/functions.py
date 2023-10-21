import numpy as np
import pandas as pd
from numba import jit
import random
import matplotlib.pyplot as plt
from pprint import pprint
from IPython.display import display
from tabulate import tabulate
import warnings
import torch
import random
import torch.nn as nn
import torch.optim as optim
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10,8) 
plt.style.use('ggplot')

def gametype_to_ran(gametype = '1236'):
    # convert game type into upper bounds for token values
    R1 = 3 ** int(gametype[0]) - 1
    R2 = 3 ** int(gametype[1]) - 1
    R3 = 3 ** int(gametype[2]) - 1
    R4 = 3 ** int(gametype[3]) - 1
    return R1, R2, R3, R4

def token_init(ntokens, nbuyers, nsellers, R1, R2, R3, R4):
    # generate values
    A = np.random.uniform(0, R1)  # common to all
    B = np.random.uniform(0, R2, (2, ))  # buyer/seller differentiator
    C = np.random.uniform(0, R3, (2, ntokens)) # buyer/seller and token differentiator
    D = np.random.uniform(0, R4, (nbuyers+nsellers, ntokens)) # unique token and trader values
    E = D + A
    
    # buyer valuations
    redemption_values = E[0:nbuyers, 0:ntokens]+C[0,:]+B[0]
    sorted_indices = np.argsort(redemption_values, axis=1)[:, ::-1]
    redemption_values = np.take_along_axis(redemption_values, sorted_indices, axis=1)

    # seller costs
    token_costs = E[nbuyers:(nbuyers+nsellers), 0:ntokens]+C[1,:]+B[1]
    sorted_indices_costs = np.argsort(token_costs, axis=1)
    token_costs = np.take_along_axis(token_costs, sorted_indices_costs, axis=1)
    return np.round(redemption_values, 1), np.round(token_costs, 1)

def compute_demand_supply(redemption_values,token_costs,nbuyers,ntokens,granularity=100):
    max_price = np.max(redemption_values)
    min_price = np.min(token_costs)
    P_grid = np.linspace(min_price,max_price,granularity)
    demand_schedule = np.zeros((granularity),dtype = 'int')
    supply_schedule = np.zeros((granularity), dtype = 'int')
    for i, p in enumerate(P_grid):
        demand_schedule[i] = np.sum(p<=redemption_values)  
        supply_schedule[i] = np.sum(p>=token_costs) 
    return demand_schedule, supply_schedule, P_grid, min_price, max_price

def equilibrium(demand_schedule,supply_schedule,P_grid):
    p_eqbm, q_eqbm = [], np.nan
    for i, p in enumerate(P_grid):
        if demand_schedule[i] == supply_schedule[i]: # when sellers are ready to sell
            p_eqbm.append(p)
            q_eqbm = demand_schedule[i] 
    return np.nanmean(p_eqbm), q_eqbm


def graph(demand_schedule, supply_schedule, P_grid, p_eqbm, q_eqbm, 
                        period_bids, period_asks, period_prices, period_sales, 
                        redemption_values, token_costs, ntokens, nbuyers, nsellers, nsteps):
    plt.plot(demand_schedule, P_grid, label = 'Demand Curve')
    plt.plot(supply_schedule, P_grid, label = 'Supply Curve')
    plt.plot(period_prices, color='green', linestyle='--', label='Mean Real Prices')
    plt.axhline(y=np.nanmean(p_eqbm), color='black', linestyle='--', label='Mean Eqbm Prices')
    prices = []
    for i in range(nsteps):
        if (period_prices[i] > 0):
            plt.scatter([period_sales[i]]*len(period_bids[i]), period_bids[i], s = 10, marker = 'x', c = 'red')
            plt.scatter([period_sales[i]]*len(period_asks[i]), period_asks[i], s = 10, marker = 'o', c = 'blue')
        else:
            pass  
    plt.legend()
    text_content = f'q*={q_eqbm}, mean(q)={np.round(np.nanmax(period_sales),1)},p*={np.round(p_eqbm,1)}, mean(p)={np.round(np.nanmean(period_prices),1)}'
    plt.title(text_content)
    plt.show()
    
    plt.plot(period_bids, c = 'r', linestyle='--')
    plt.plot(period_asks, c = 'b', linestyle='--')
    plt.scatter(range(nsteps), period_prices, c = 'g')
    plt.title('Bids (red), Asks (blue) and Prices (green) over trading steps')
    plt.show()

def graph_period(db, rnd, period):
    period_bids = list(db.get_period(rnd, period).bids)
    period_asks = list(db.get_period(rnd, period).asks)
    period_prices = list(db.get_period(rnd, period).price)
    period_sales = np.cumsum(np.where(db.get_period(rnd, period).price > 0,1,0))
    [_, demand_schedule, supply_schedule, P_grid, redemption_values, token_costs, p_eqbm, q_eqbm] = db.get_round(rnd).iloc[0].tolist()
    graph(demand_schedule, supply_schedule, P_grid, p_eqbm, q_eqbm, period_bids, period_asks, period_prices, period_sales, redemption_values, token_costs, db.ntokens, db.nbuyers, db.nsellers, db.nsteps)
    
    
def current_bid_ask(bids, asks):
    if np.all(np.isnan(bids)) == False:
        current_bid_idx = np.nanargmax(bids)
        current_bid = np.nanmax(bids)
    else:
        current_bid_idx = np.nan
        current_bid = np.nan

    if np.all(np.isnan(asks)) == False:
        current_ask_idx = np.nanargmin(asks)
        current_ask = np.nanmin(asks)
    else:
        current_ask_idx = np.nan
        current_ask = np.nan
    return current_ask, current_ask_idx, current_bid, current_bid_idx

def buy_sell(db, current_bid, current_bid_idx, current_ask, current_ask_idx):
        sale, price, bprofit, sprofit = 0, np.nan, 0, 0 
        if (current_bid_idx >= 0) and (current_ask_idx >= 0):
            buy = db.buyers[current_bid_idx].buy(current_bid,current_ask)
            sell = db.sellers[current_ask_idx].sell(current_bid,current_ask)
            if buy and not sell:
                price = current_ask
            elif sell and not buy:
                price = current_bid
            elif sell and buy: 
                #price = np.random.choice([current_bid,current_ask])
                price = (current_bid + current_ask)/2
            else:
                price = np.nan

            if price > 0:
                db.buyers[current_bid_idx].transact(price)
                db.sellers[current_ask_idx].transact(price)
                sale = 1
                bprofit = db.buyers[current_bid_idx].step_profit
                sprofit = db.sellers[current_ask_idx].step_profit
            return sale, price, bprofit, sprofit, buy, sell
        else:
            return 0, np.nan, 0, 0, np.nan, np.nan 
    
    
class Database:
    
    def __init__(self, game_metadata, buyer_strategies, seller_strategies):
        self.nrounds, self.nperiods, self.ntokens, self.nbuyers, self.nsellers, self.nsteps, self.R1, self.R2, self.R3, self.R4 = game_metadata
        self.step_data = pd.DataFrame(columns=['rnd', 'period', 'step', 'bids','asks','current_bid','current_bid_idx','current_ask','current_ask_idx','buy','sell','price','sale', 'bprofit', 'sprofit'])
        self.round_data = pd.DataFrame(columns=['rnd', 'demand_schedule', 'supply_schedule', 'P_grid', 'redemption_values', 'token_costs','p_eqbm', 'q_eqbm'])
        self.buyers, self.sellers = generate_agents(game_metadata, buyer_strategies, seller_strategies)
        
    def add_step(self, data):
        self.step_data.loc[len(self.step_data.index)] = data
    
    def get_period(self, rnd, period):
        temp = self.step_data[(self.step_data.rnd==rnd) & (self.step_data.period==period)]
        temp = temp[['step', 'bids', 'asks', 'current_bid', 'current_bid_idx', 'current_ask','current_ask_idx','buy','sell','price', 'bprofit', 'sprofit']]
        return temp
    
    def reset_period(self, rnd):
        for i in range(self.nbuyers):
            self.buyers[i].reset(self, rnd)
        for i in range(self.nsellers):
            self.sellers[i].reset(self, rnd)

    def reset_round(self, rnd, ntokens, nbuyers, nsellers, R1, R2, R3, R4):
        redemption_values, token_costs = token_init(ntokens, nbuyers, nsellers, R1, R2, R3, R4)
        demand_schedule, supply_schedule, P_grid, min_price, max_price = compute_demand_supply(redemption_values,token_costs, self.nbuyers, self.ntokens)
        p_eqbm, q_eqbm = equilibrium(demand_schedule,supply_schedule,P_grid) 
        self.round_data.loc[len(self.round_data.index)] = [rnd, demand_schedule, supply_schedule, P_grid, redemption_values, token_costs, p_eqbm, q_eqbm]
    
    def get_round(self, rnd):
        temp = self.round_data[(self.round_data.rnd==rnd)]
        temp = temp[['rnd', 'demand_schedule', 'supply_schedule', 'P_grid', 'redemption_values', 'token_costs','p_eqbm', 'q_eqbm']]
        return temp
    
class Trader:
    def __init__(self, game_metadata, name, buyer=True):
        self.name, self.buyer, self.index, self.profit_history = name, buyer, int(name[1]), []
        self.nrounds, self.nperiods, self.ntokens, self.nbuyers, self.nsellers, self.nsteps, self.R1, self.R2, self.R3, self.R4, = game_metadata 
         
    def reset(self, db, rnd):
        round_data = db.get_round(rnd)
        self.num_tokens_traded = 0
        if self.buyer == True:
            self.token_values = list(round_data.redemption_values.item()[self.index, :])
        else:
            self.token_values = list(round_data.token_costs.item()[self.index, :])

    def transact(self, price):
        self.num_tokens_traded += 1
        if self.buyer == True:
            self.step_profit = self.value-price
        else:
            self.step_profit = price-self.value
    
    def next_token(self):
        if self.num_tokens_traded == self.ntokens:
            self.value = np.nan
        else:
            self.value = self.token_values[self.num_tokens_traded]
    
    def buy(self, current_bid, current_ask):
        if self.value >= current_ask:
            return True
        else:
            return False 
    
    def sell(self, current_bid, current_ask):
        if current_bid >= self.value:
            return True
        else:
            return False
    
    def learn(self, db):
        pass
    
    def show_avg_profit(self):
        return 
    
class Honest(Trader):
    def __init__(self, game_metadata, name, buyer=True):
        super().__init__(game_metadata, name, buyer)
    
    def bid(self, db):
        self.next_token()
        self.bid_amount = self.value
        return self.bid_amount
    
    def ask(self, db):
        self.next_token()
        self.ask_amount = self.value
        return self.ask_amount
    
class Random(Trader):
    def __init__(self, game_metadata, name, buyer=True):
        super().__init__(game_metadata, name, buyer)

    def bid(self, db):
        self.next_token()
        if self.value != np.nan:
            self.bid_amount = np.round(np.random.uniform(0.05*self.value, self.value, 1).item(),1)
        else:
            self.bid_amount = np.nan
        return self.bid_amount
    
    def ask(self, db):
        self.next_token()
        if self.value != np.nan:
            self.ask_amount = np.round(np.random.uniform(self.value, self.value*1.9, 1).item(),1)
        else:
            self.ask_amount = np.nan
        return self.ask_amount
    
def generate_agents(game_metadata,buyer_strategies,seller_strategies):
    buyers = []
    for idx,i in enumerate(buyer_strategies):
        if i == 'Honest':
            buyers.append(Honest(game_metadata,'B'+str(idx),buyer=True)) 
        if i == 'DQN':
            buyers.append(DQN(game_metadata,'B'+str(idx),buyer=True)) 
        if i == 'Random':
            buyers.append(Random(game_metadata,'B'+str(idx),buyer=True)) 
                        
    sellers = []
    for idx,i in enumerate(seller_strategies):
        if i == 'Honest':
            sellers.append(Honest(game_metadata,'S'+str(idx),buyer=False))    
        if i == 'Random':
            sellers.append(Random(game_metadata,'S'+str(idx),buyer=False))
    return buyers, sellers