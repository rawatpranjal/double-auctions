
from packages import *
from setup import *
from database import *
from traders import *

def TradingGame(buyer_strategies, seller_strategies, verbose = 0):
    #nrounds, nperiods, ntokens, R1, R2, R3, R4, minprice, maxprice, nsteps = game_setup()
    nrounds, nperiods, ntokens, R1, R2, R3, R4, minprice, maxprice, nsteps = 1, 1, 4, 10, 10, 10, 10, 1, 100, 25
    nbuyers, nsellers = len(buyer_strategies), len(seller_strategies)
    game_metadata = [nrounds, nperiods, ntokens, nbuyers, nsellers, R1, R2, R3, R4, minprice, maxprice, nsteps]
    buyers, sellers = generate_agents(buyer_strategies, seller_strategies, game_metadata)
    db = Database(game_metadata, buyers, sellers)

    # Begin game
    for rnd in range(nrounds):

        # assign token values for this round
        token_values = round_setup(ntokens, nbuyers, nsellers, R1, R2, R3, R4)
        
        # store round data
        p_eqbm, q_eqbm = compute_demand_supply(token_values, ntokens, nbuyers,nsellers)
        db.add_round([rnd, token_values, p_eqbm, q_eqbm])
        try:
            p_eqbm = np.nanmean(p_eqbm)
        except:
            p_eqbm = np.nan
        
        for period in range(nperiods):
            
            # reset period
            for i in range(nbuyers):
                db.buyers[i].reset(token_values[i])
            for i in range(nsellers):
                db.sellers[i].reset(token_values[i])
            
            for step in range(nsteps):

                # bid/ask step
                bids = [buyer.bid(db) for buyer in db.buyers]
                asks = [seller.ask(db) for seller in db.sellers]

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

                # buy/sell step 
                sale = 0
                price = np.nan
                bprofit = np.nan
                sprofit = np.nan
                buy = 0
                sell = 0
                bprofit = 0
                sprofit = 0
                try:
                    if db.buyers[current_bid_idx].buy(current_bid,current_ask):
                        buy = 1
                    if db.sellers[current_ask_idx].sell(current_bid,current_ask):
                        sell = 1
                    if buy == 1 and sell == 1:
                        #price = np.random.choice([current_bid,current_ask])
                        price = (current_bid + current_ask)/2
                        db.buyers[current_bid_idx].transact(price)
                        db.sellers[current_ask_idx].transact(price)
                        sale = 1
                        bprofit = db.buyers[current_bid_idx].step_profit
                        sprofit = db.sellers[current_ask_idx].step_profit
                except:
                    buy = np.nan
                    sell = np.nan
                    
                # store step data
                step_data = [rnd,period,step,bids,asks,current_bid,current_bid_idx,current_ask,current_ask_idx,buy,sell,price,sale,bprofit,sprofit, p_eqbm, q_eqbm]
                db.add_step(step_data)
        
    return db