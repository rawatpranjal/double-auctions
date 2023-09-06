#!/usr/bin/env python3

from packages import *

########################################################################
######################## Initial Conditions
########################################################################

def game_setup(max_rounds = 6, max_periods = 6, max_tokens = 6, max_K = 6, minprice = 1, maxprice = 2000, nsteps = 25):
    
    # generate num rounds, num periods and num tokens
    nrounds = np.random.randint(1,max_rounds)
    nperiods = np.random.randint(1,max_periods)
    ntokens = np.random.randint(1,max_tokens)
    
    # generate R1, R2, R3, R4 which will define token distribution
    gametype_1 = np.random.randint(1,max_K)
    gametype_2 = np.random.randint(1,max_K)
    gametype_3 = np.random.randint(1,max_K)
    gametype_4 = np.random.randint(1,max_K)
    R1 = 3 ** gametype_1 - 1
    R2 = 3 ** gametype_2 - 1
    R3 = 3 ** gametype_3 - 1
    R4 = 3 ** gametype_4 - 1
    
    return nrounds, nperiods, ntokens, R1, R2, R3, R4, minprice, maxprice, nsteps

def round_setup(ntokens, nbuyers, nsellers, R1, R2, R3, R4):
    
    # draw a set of token value from give token distribution
    A = np.random.uniform(0,R1)
    B = np.random.uniform(0,R2)
    C = np.random.uniform(0,R3,(nbuyers + nsellers,1))
    D = np.random.uniform(0,R4,(nbuyers + nsellers,ntokens))
    token_values = A + B + C + D
    
    # sort by row
    sorted_indices = np.argsort(token_values, axis=1)[:, ::-1]
    token_values = np.take_along_axis(token_values, sorted_indices, axis=1)
    return np.round(token_values,1)

########################################################################
######################## Compute Equilibrium
########################################################################

@jit
def token_value_breakup(token_values,nbuyers,nsellers):
    
    # break tokens between buyers and sellers
    buyer_token_values = token_values[0:nbuyers,:]
    seller_token_values = token_values[nbuyers:nbuyers+nsellers,:]
    
    # obtain minimum and maximum prices
    max_eqbm_price = np.max(buyer_token_values)
    min_eqbm_price = np.min(seller_token_values)
    return buyer_token_values, seller_token_values, max_eqbm_price, min_eqbm_price

@jit
def compute_demand_curve(buyer_token_values,nbuyers,ntokens,min_eqbm_price,max_eqbm_price,granularity=1000):
    
    # create a grid of price based on minimum and maximum price
    P_grid = np.linspace(min_eqbm_price,max_eqbm_price,granularity)
    
    # find number of units that would be demanded as price rises
    demand_schedule = np.zeros((granularity),dtype = 'int')
    for i, p in enumerate(P_grid):
        demand = np.sum(p<buyer_token_values) # how many tokens are buyers ready to buy at this price
        demand_schedule[i] = demand
        
    return demand_schedule, P_grid

@jit
def compute_supply_curve(seller_token_values,nsellers,ntokens,min_eqbm_price,max_eqbm_price,granularity=1000):
    
    # create a grid of price based on minimum and maximum price
    P_grid = np.linspace(min_eqbm_price,max_eqbm_price,granularity)
    
    # find number of units that would be supplied as price rises
    supply_schedule = np.zeros((granularity), dtype = 'int')
    for i, p in enumerate(P_grid):
        supply = np.sum(p>seller_token_values) # how much sellers are ready to sell at this price
        supply_schedule[i] = supply
        
    return supply_schedule, P_grid

@jit
def find_equilibrium(demand_schedule,supply_schedule,P_grid):
    
    # create price equilibrium placeholders
    p_eqbm = []
    q_eqbm = np.nan
    
    # find prices at which demand and supply clears
    for i, p in enumerate(P_grid):
        if demand_schedule[i] == supply_schedule[i]: # when sellers are ready to sell
            p_eqbm.append(p)
            q_eqbm = demand_schedule[i]
            
    return p_eqbm, q_eqbm

def compute_demand_supply(token_values, ntokens, nbuyers, nsellers):
    
    # split buyer and seller tokens
    buyer_token_values, seller_token_values, max_eqbm_price, min_eqbm_price = token_value_breakup(token_values,nbuyers,nsellers)
    
    # find demand curve
    demand_schedule, P_grid = compute_demand_curve(buyer_token_values,nbuyers,ntokens,min_eqbm_price,max_eqbm_price)
    
    # find supply curve
    supply_schedule, P_grid = compute_supply_curve(seller_token_values,nsellers,ntokens,min_eqbm_price,max_eqbm_price)
    
    # compute equilibrium
    p_eqbm, q_eqbm = find_equilibrium(demand_schedule,supply_schedule,P_grid)
    
    return p_eqbm, q_eqbm

########################################################################
######################## Plotting
########################################################################

def plot_period_results(period_bids, period_asks, period_prices, period_sales, token_values, ntokens, nbuyers, nsellers, nsteps, R1, R2, R3, R4):
    
    # compute equilibrium
    buyer_token_values, seller_token_values, max_eqbm_price, min_eqbm_price = token_value_breakup(token_values,nbuyers,nsellers)
    demand_schedule, P_grid = compute_demand_curve(buyer_token_values,nbuyers,ntokens,min_eqbm_price,max_eqbm_price)
    supply_schedule, P_grid = compute_supply_curve(seller_token_values,nsellers,ntokens,min_eqbm_price,max_eqbm_price)
    p_eqbm, q_eqbm = find_equilibrium(demand_schedule,supply_schedule,P_grid)
    
    # plot demand and supply schedules
    plt.plot(demand_schedule, P_grid, label = 'Demand Curve')
    plt.plot(supply_schedule, P_grid, label = 'Supply Curve')
    count = 0
    prices = []
    
    # plot the actual bids and asks, including actual prices
    for i in range(nsteps):
        bids = period_bids[i]
        asks = period_asks[i]
        price = period_prices[i]
        sales = period_sales[i]
        if price != np.nan:
            plt.scatter([sales]*len(bids), bids, s = 10, marker = 'x', c = 'red')
            plt.scatter([sales]*len(asks), asks, s = 10, marker = 'o', c = 'blue')
            #plt.scatter(sales, price, s = 20, marker = '^', c = 'green')
        else:
            pass
        
    # plot average and equilibrium prices    
    plt.plot(period_prices, color='green', linestyle='--', label='Mean Real Prices')
    plt.axhline(y=np.nanmean(p_eqbm), color='black', linestyle='--', label='Mean Eqbm Prices')
    plt.legend()
    plt.show()
    
    # print information
    print('Eqbm quantity:', q_eqbm)
    try:
        print('Eqbm prices:', p_eqbm[0], 'to', p_eqbm[-1])
    except:
        print('No Eqbm price')
    try:
        print('Avg Prices:', np.nanmean(period_prices))
        print('Actual quantity', np.nanmax(period_sales))
    except: 
        print('No sales')
        
    # plot bid, ask, and actual prices over timesteps
    plt.plot(period_bids, c = 'r', linestyle='--')
    plt.plot(period_asks,  c = 'b', linestyle='--')
    plt.scatter(range(nsteps), period_prices, c = 'g')
    plt.title('Bids, Asks, Prices over time')
    plt.show()
    
########################################################################
######################## Outcomes
########################################################################

def individual_outcomes(db, buyers = True):
    df = db.step_data
    a = df.groupby(['current_ask_idx']).mean()[['current_ask', 'price','sprofit']]
    b = df.groupby(['current_ask_idx']).sum()[['sell','sprofit']]
    seller_outcome = pd.concat([a,b], axis = 1)
    seller_outcome.columns = ['mean_asks', 'mean_price', 'mean_profit','total_sales','total_profit']
    a = df.groupby(['current_bid_idx']).mean()[['current_bid', 'price','bprofit']]
    b = df.groupby(['current_bid_idx']).sum()[['sell','bprofit']]
    buyer_outcome = pd.concat([a,b], axis = 1)
    buyer_outcome.columns = ['mean_bids', 'mean_price', 'mean_profit','total_sales','total_profit']
    if buyers == True:
        return buyer_outcome
    else:
        return seller_outcome
    
def market_outcomes(db):
    df = db.step_data
    a = df.groupby(['rnd', 'period']).mean()[['price','p_eqbm','q_eqbm']]
    b = df.groupby(['rnd', 'period']).sum()[['sell']]
    market = pd.concat([a,b], axis = 1)
    market.columns = ['mean_market_price', 'p_eqbm', 'q_eqbm','total_sales']
    return market
    
  



