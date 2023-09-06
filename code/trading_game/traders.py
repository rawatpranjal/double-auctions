#!/usr/bin/env python3

from packages import *

########################################################################
######################## TRADERS
########################################################################

class Trader:
    # initialize agent at beginning of game with public information
    def __init__(self, public_information, name, buyer=True):
        self.name, self.buyer = name, buyer
        self.step_profit, self.period_profit, self.round_profit, self.game_profit, self.bids_on_this_token = 0, 0, 0, 0, 0
        self.nrounds, self.nperiods, self.ntokens, self.nbuyers, self.nsellers, self.R1, self.R2, self.R3, self.R4, self.minprice, self.maxprice, self.nsteps = public_information 
        
    # at start of round, add the private information
    def reset_round(self, token_values):
        self.step_profit, self.period_profit, self.round_profit = 0, 0, 0
        self.token_values = list(np.round(np.sort(token_values, kind='quicksort')[::-1],1))
        if self.buyer == True:
            self.num_tokens_held = 0
            self.tokens_held = []
            self.bids_on_this_token = 0
        else:
            self.num_tokens_held = len(self.token_values)
            self.tokens_held = self.token_values

    # at end of period, reset data
    def reset_period(self, token_values):
        self.step_profit, self.period_profit, self.round_profit = 0, 0, 0
        self.token_values = list(np.round(np.sort(token_values, kind='quicksort')[::-1],1))
        if self.buyer == True:
            self.num_tokens_held = 0
            self.tokens_held = []
            self.period_profit = 0
            self.bids_on_this_token = 0
        else:
            self.num_tokens_held = len(self.token_values)
            self.tokens_held = self.token_values
            self.period_profit = 0
    
    # make the transaction
    def transact(self, price):
        if self.buyer == True:
            self.num_tokens_held += 1
            self.tokens_held.append(self.value)
            self.step_profit = self.value-price
            self.bids_on_this_token = 0
        else:
            self.num_tokens_held = self.num_tokens_held -1
            self.tokens_held.remove(self.value)
            self.step_profit = price-self.value

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
        if self.num_tokens_held == len(self.token_values):
            return np.nan
        n = self.num_tokens_held + 1 # index of token
        self.value = np.partition(self.token_values, -n)[-n] # get nth max
        self.bid_amount = np.round(np.random.uniform(0.8*self.value, self.value, 1).item(),1)
        return self.bid_amount
    
    def ask(self, db):
        if self.num_tokens_held == 0:
            return np.nan
        n = self.num_tokens_held
        self.value = np.partition(self.token_values, -n)[-n]
        self.ask_amount = np.round(np.random.uniform(self.value, 1.2*self.value, 1).item(),1)
        return self.ask_amount
    
    def buy(self, current_bid, current_ask):
        if current_ask <= current_bid:
            return True
        else:
            return False
        
    def sell(self, current_bid, current_ask):
        if current_ask <= current_bid:
            return True
        else:
            return False
        
########################################################################
######################## HONEST
########################################################################
        
class Honest(Trader):
    def __init__(self, public_information, name, buyer=True):
        super().__init__(public_information, name, buyer)

    def bid(self, db):
        if self.num_tokens_held == len(self.token_values):
            return np.nan
        n = self.num_tokens_held + 1 # index of token
        self.value = np.partition(self.token_values, -n)[-n] # get nth max
        self.bid_amount = self.value
        return self.bid_amount
    
    def ask(self, db):
        if self.num_tokens_held == 0:
            return np.nan
        n = self.num_tokens_held
        self.value = np.partition(self.token_values, -n)[-n]
        self.ask_amount = self.value
        return self.ask_amount
    
    def buy(self, current_bid, current_ask):
        if current_ask <= current_bid:
            return True
        else:
            return False
        
    def sell(self, current_bid, current_ask):
        if current_ask <= current_bid:
            return True
        else:
            return False
        
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
    
    def buy(self, current_bid, current_ask):
        if current_ask <= current_bid:
            return True
        else:
            return False
        
    def sell(self, current_bid, current_ask):
        if current_ask <= current_bid:
            return True
        else:
            return False
        
        
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
        return self.bid_amount

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
        return self.ask_amount

    def buy(self, current_bid, current_ask):
        if current_ask <= current_bid:
            return True
        else:
            return False

    def sell(self, current_bid, current_ask):
        if current_ask <= current_bid:
            return True
        else:
            return False
        
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
        return self.bid_amount

    def ask(self, db):
        if self.num_tokens_held == 0:
            return np.nan
        n = self.num_tokens_held
        self.value = np.partition(self.token_values, -n)[-n]
        self.ask_amount = self.value*1.1
        return self.ask_amount

    def buy(self, current_bid, current_ask):
        if current_ask <= current_bid:
            return True
        else:
            return False

    def sell(self, current_bid, current_ask):
        if current_ask <= current_bid:
            return True
        else:
            return False      
        
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
                     
    return buyers, sellers