#!/usr/bin/env python3

from packages import *
from setup import *

########################################################################
######################## TRADERS
########################################################################

class Trader:
    # initialize agent at beginning of game with public information
    def __init__(self, public_information, name, buyer=True):
        self.name, self.buyer = name, buyer
        self.step_profit, self.period_profit, self.round_profit, self.game_profit, self.bids_on_this_token = 0, 0, 0, 0, 0
        self.nrounds, self.nperiods, self.ntokens, self.nbuyers, self.nsellers, self.R1, self.R2, self.R3, self.R4, self.minprice, self.maxprice, self.nsteps = public_information 
        
    # reset trader with token values
    def reset(self, token_values):
        self.step_profit = 0
        self.token_values = list(token_values)
        if self.buyer == True:
            self.num_tokens_held = 0 # tokens held
            self.num_tokens_traded = 0
            self.bids_on_this_token = 0
        else:
            self.num_tokens_held = len(token_values)-1
            self.num_tokens_traded = 0
            self.asks_on_this_token = 0

    # make the transaction
    def transact(self, price):
        self.num_tokens_traded += 1
        if self.buyer == True:
            self.num_tokens_held += 1
            self.step_profit = self.value-price
            self.bids_on_this_token = 0
        else:
            self.num_tokens_held -= 1
            self.step_profit = price-self.value
            self.asks_on_this_token = 0
    
    # find valuation of next token, nan if no valuation
    def next_token(self):
        if self.buyer == True:
            if self.num_tokens_held == self.ntokens:
                self.value = np.nan
            else:
                self.value = self.token_values[self.num_tokens_held]
        else:
            if self.num_tokens_held == 0:
                self.value = np.nan
            else:
                self.value = self.token_values[self.num_tokens_held]

    # buy if anticipated price (avg of current bid and ask) is less than own valuation
    def buy(self, current_bid, current_ask):
        if self.value >= (current_bid + current_ask)/2:
            return True
        else:
            return False
        
    # sell if anticipated price (avg of current bid and ask) is more than own valuation
    def sell(self, current_bid, current_ask):
        if (current_bid + current_ask)/2 >= self.value:
            return True
        else:
            return False
        
    def describe(self):
        print(f"\n")
        print(f"Name: {self.name}")
        print(f"Buyer: {self.buyer}")
        print(f"Tokens Held: {self.tokens_held}")
        print(f"Token Values: {self.token_values}")
        print(f"Period Profit: {self.period_profit}")
        print(f"Round Profit: {self.round_profit}")
        print(f"Game Profit: {self.game_profit}")
        
########################################################################
######################## RANDOM
########################################################################
    
class Random(Trader):
    def __init__(self, public_information, name, buyer=True):
        super().__init__(public_information, name, buyer)

    def bid(self, db):
        self.next_token()
        if self.value != np.nan:
            self.bid_amount = np.round(np.random.uniform(0.7*self.value, self.value, 1).item(),1)
        else:
            self.bid_amount = np.nan
        return self.bid_amount
    
    def ask(self, db):
        self.next_token()
        if self.value != np.nan:
            self.ask_amount = np.round(np.random.uniform(self.value, self.value*1.2, 1).item(),1)
        else:
            self.ask_amount = np.nan
        return self.ask_amount
        
########################################################################
######################## HONEST
########################################################################
        
class Honest(Trader):
    def __init__(self, public_information, name, buyer=True):
        super().__init__(public_information, name, buyer)

    def bid(self, db):
        self.next_token()
        self.bid_amount = self.value
        return self.bid_amount
    
    def ask(self, db):
        self.next_token()
        self.ask_amount = self.value
        return self.ask_amount
    
########################################################################
######################## CREEPER
########################################################################

class Creeper(Trader):
    def __init__(self, public_information, name, buyer=True):
        super().__init__(public_information, name, buyer)

    def bid(self, db):
        if (self.num_tokens_held == len(self.token_values)):
            return np.nan
        n = self.num_tokens_held + 1 # index of token
        self.value = np.partition(self.token_values, -n)[-n] # get nth max
        if self.bids_on_this_token == 0:
            self.bid_amount = self.value*0.7
        else:
            data = db.step_data.iloc[-1]
            if data.sale == 0:
                self.bid_amount = np.min([self.value,self.bid_amount+0.3*(self.value - self.bid_amount)]) 
        self.bids_on_this_token += 1
        return np.round(self.bid_amount,1)
    
    def ask(self, db):
        if self.num_tokens_held == 0:
            return np.nan
        n = self.num_tokens_held
        self.value = np.partition(self.token_values, -n)[-n] # get nth max
        try: 
            data = db.step_data.iloc[-1]
            if (data.step > 0):
                if (data.current_ask_idx != int(self.name[-1])):
                    self.ask_amount += 0.2*(self.value - self.ask_amount) # decrease if didn't get current_ask
            else:
                self.ask_amount = self.value*1.3 # ask at start of the game
        except: 
            self.ask_amount = self.value*1.3
        return np.round(self.ask_amount,1)
        
########################################################################
######################## SNIPER
########################################################################

class Sniper(Trader):
    def __init__(self, public_information, name, buyer=True):
        super().__init__(public_information, name, buyer)

    def bid(self, db):
        if (self.num_tokens_held == len(self.token_values)):
            return np.nan
        n = self.num_tokens_held + 1 # index of token
        self.value = np.partition(self.token_values, -n)[-n] # get nth max
        self.bid_amount = np.nan
        try: 
            data = db.step_data.iloc[-1]
            if (data.step > 0):
                if (data.current_bid > 0.9 * data.current_ask):
                    self.bid_amount = data.current_ask
        except: 
            pass
        return np.round(self.bid_amount,1)

    def ask(self, db):
        if self.num_tokens_held == 0:
            return np.nan
        n = self.num_tokens_held
        self.value = np.partition(self.token_values, -n)[-n]
        self.ask_amount = np.nan
        try: 
            data = db.step_data.iloc[-1]
            if (data.step > 0):
                if (data.current_bid > 0.9 * data.current_ask):
                    self.ask_amount = data.current_bid
        except: 
            pass
        return np.round(self.ask_amount,1)

        
########################################################################
######################## MARKUP
########################################################################

class Markup(Trader):
    def __init__(self, public_information, name, buyer=True):
        super().__init__(public_information, name, buyer)

    def bid(self, db):
        if (self.num_tokens_held == len(self.token_values)):
            return np.nan
        n = self.num_tokens_held + 1 # index of token
        self.value = np.partition(self.token_values, -n)[-n] # get nth max
        self.bid_amount = self.value*0.9
        return np.round(self.bid_amount,1)

    def ask(self, db):
        if self.num_tokens_held == 0:
            return np.nan
        n = self.num_tokens_held
        self.value = np.partition(self.token_values, -n)[-n]
        self.ask_amount = self.value*1.1
        return np.round(self.ask_amount,1)
    
########################################################################
######################## FORESIGHT
########################################################################
    
class Foresight(Trader):
    def __init__(self, public_information, name, buyer=True):
        super().__init__(public_information, name, buyer)

    def bid(self, db):
        if self.num_tokens_held == len(self.token_values):
            return np.nan
        n = self.num_tokens_held
        self.value = np.partition(self.token_values, -n)[-n]
        try: 
            data = db.step_data.iloc[-1]
            rnd = data.rnd
            token_values = db.get_token_values(rnd).item()
            p_eqbm, q_eqbm = compute_demand_supply(token_values, self.ntokens, self.nbuyers, self.nsellers)
            p_eqbm = np.nanmean(p_eqbm)
            if (p_eqbm != np.nan) & (p_eqbm <= self.value):
                    self.bid_amount = p_eqbm
        except: 
            print('error')
            self.bid_amount = np.nan
        return np.round(self.bid_amount,1)
    
    def ask(self, db):
        if self.num_tokens_held == 0:
            return np.nan
        n = self.num_tokens_held
        self.value = np.partition(self.token_values, -n)[-n]
        try: 
            data = db.step_data.iloc[-1]
            rnd = data.rnd
            token_values = db.get_token_values(rnd).item()
            p_eqbm, q_eqbm = compute_demand_supply(token_values, self.ntokens, self.nbuyers, self.nsellers)
            p_eqbm = np.nanmean(p_eqbm)
            if (p_eqbm != np.nan) & (p_eqbm >= self.value):
                    self.ask_amount = p_eqbm
        except: 
            self.ask_amount = np.nan
        return np.round(self.ask_amount,1)
         
        
########################################################################
######################## GENERATE
########################################################################

def generate_agents(buyer_strategies, seller_strategies, public_information):
    
    # populate a set of buyers
    buyers = []
    for idx,i in enumerate(buyer_strategies):
        if i == 'Honest':
            buyers.append(Honest(public_information,'B'+str(idx),buyer=True))   
        if i == 'Random':
            buyers.append(Random(public_information,'B'+str(idx),buyer=True))   
        if i == 'Sniper':
            buyers.append(Sniper(public_information,'B'+str(idx),buyer=True))   
        if i == 'Creeper':
            buyers.append(Creeper(public_information,'B'+str(idx),buyer=True))   
        if i == 'Markup':
            buyers.append(Markup(public_information,'B'+str(idx),buyer=True))   
        if i == 'Foresight':
            buyers.append(Foresight(public_information,'B'+str(idx),buyer=True))   
                                                                           
    # populate a set of sellers
    sellers = []
    for idx,i in enumerate(seller_strategies):
        if i == 'Honest':
            sellers.append(Honest(public_information,'S'+str(idx),buyer=False))   
        if i == 'Random':
            sellers.append(Random(public_information,'S'+str(idx),buyer=False))   
        if i == 'Sniper':
            sellers.append(Sniper(public_information,'S'+str(idx),buyer=False))   
        if i == 'Creeper':
            sellers.append(Creeper(public_information,'S'+str(idx),buyer=False))   
        if i == 'Markup':
            sellers.append(Markup(public_information,'S'+str(idx),buyer=False))   
        if i == 'Foresight':
            sellers.append(Foresight(public_information,'S'+str(idx),buyer=False))   
                                     
    return buyers, sellers