import torch
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train import MarketDN

H5_FILE = "../data/kraken_lob_data.h5"
MODEL_FILE = "../models/market_gnn.pth"
INITIAL_BALANCE = 1000.0 
FEE_RATE = 0.0026        
CONFIDENCE_THRESHOLD = 0.60 

# Dummy Batch Class for the Model
class MiniBatch:
    def __init__(self, x):
        self.x = x
        self.num_graphs = 1

def backtest():
    # Load the Brain
    device = torch.device('cpu') 
    print(f"--- LOADING BRAIN ({device}) ---")
    
    model = MarketDN().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    # Portfolio State
    usd_balance = INITIAL_BALANCE
    btc_balance = 0.0
    
    portfolio_history = []
    price_history = []
    
    # Market State
    mid_price_history = []
    bids = {} 
    asks = {} 
    
    print("--- STARTING SIMULATION ---")
    
    with h5py.File(H5_FILE, 'r') as f:
        raw_json = f['raw_json'][:]
        total_ticks = len(raw_json)
        
        for i in tqdm(range(total_ticks)):
            try:
                line = raw_json[i].decode('utf-8')
                data = json.loads(line)
                
                if not isinstance(data, list): continue
                payload = data[1]

                if 'bs' in payload or 'as' in payload:
                    bids = {float(x[0]): float(x[1]) for x in payload.get('bs', [])}
                    asks = {float(x[0]): float(x[1]) for x in payload.get('as', [])}
                elif 'b' in payload or 'a' in payload:
                    for item in payload.get('b', []):
                        p = float(item[0])
                        v = float(item[1])
                        if v == 0: bids.pop(p, None)
                        else: bids[p] = v
                        
                    for item in payload.get('a', []):
                        p = float(item[0])
                        v = float(item[1])
                        if v == 0: asks.pop(p, None)
                        else: asks[p] = v

                # Valid Book Check
                if len(bids) < 10 or len(asks) < 10: continue

                # Sort
                sorted_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:10]
                sorted_asks = sorted(asks.items(), key=lambda x: x[0])[:10]
                
                best_bid = sorted_bids[0][0]
                best_ask = sorted_asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                
                # History Check
                mid_price_history.append(mid_price)
                if len(mid_price_history) > 10: mid_price_history.pop(0)
                if len(mid_price_history) < 10: continue 

                volatility = np.std(mid_price_history)
                momentum = mid_price - mid_price_history[0]
                spread = best_ask - best_bid
                total_bid_vol = sum(v for p, v in sorted_bids)
                total_ask_vol = sum(v for p, v in sorted_asks)
                imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-5)

                feature_vector = []
                for p, v in sorted_bids:
                    norm_p = (p - mid_price) / mid_price * 1000
                    log_v = np.log(v + 1.0)
                    feature_vector.extend([norm_p, log_v, -1.0, imbalance, spread, momentum, volatility])
                for p, v in sorted_asks:
                    norm_p = (p - mid_price) / mid_price * 1000
                    log_v = np.log(v + 1.0)
                    feature_vector.extend([norm_p, log_v, 1.0, imbalance, spread, momentum, volatility])

                x = torch.tensor([feature_vector], dtype=torch.float).to(device)
                
                with torch.no_grad():
                    out = model(MiniBatch(x))
                    probs = torch.exp(out) 

                p_down = probs[0][0].item()
                p_up   = probs[0][2].item()
                
                # BUY
                if p_up > CONFIDENCE_THRESHOLD and usd_balance > 10:
                    btc_bought = (usd_balance / best_ask) * (1 - FEE_RATE)
                    btc_balance += btc_bought
                    usd_balance = 0
                    
                # SELL
                elif p_down > CONFIDENCE_THRESHOLD and btc_balance > 0.0001:
                    usd_received = (btc_balance * best_bid) * (1 - FEE_RATE)
                    usd_balance += usd_received
                    btc_balance = 0

                # Track Value
                current_value = usd_balance + (btc_balance * mid_price)
                portfolio_history.append(current_value)
                price_history.append(mid_price)

            except Exception as e:
                # print(e)
                continue

    if len(portfolio_history) > 0:
        final_value = portfolio_history[-1]
        profit_pct = ((final_value - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        
        print("\n" + "="*40)
        print(f" INITIAL BALANCE: ${INITIAL_BALANCE:.2f}")
        print(f" FINAL BALANCE:   ${final_value:.2f}")
        print(f" TOTAL PROFIT:    {profit_pct:.2f}%")
        print("="*40 + "\n")

        # Plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:blue'
        ax1.set_xlabel('Time (Ticks)')
        ax1.set_ylabel('Portfolio Value ($)', color=color)
        ax1.plot(portfolio_history, color=color, label='AI Portfolio')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:gray'
        ax2.set_ylabel('BTC Price ($)', color=color)  
        ax2.plot(price_history, color=color, alpha=0.3, label='BTC Price')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('AI Strategy Backtest')
        plt.show()
    else:
        print("Error: No trades or data points recorded. Check data file.")

if __name__ == "__main__":
    backtest()