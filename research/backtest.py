import torch
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train import MarketDN

# CONFIGURATION
H5_FILE = "../data/kraken_volatility.h5"
MODEL_FILE = "../models/market_gnn.pth"
INITIAL_BALANCE = 1000.0 
MAKER_FEE = 0.000      
CONFIDENCE_THRESHOLD = 0.35 

class MiniBatch:
    def __init__(self, x):
        self.x = x
        self.num_graphs = 1

def backtest():
    device = torch.device('cpu') 
    print(f"--- LOADING BRAIN ({device}) ---")
    
    # Initialize Model
    model = MarketDN()
    model.input_dim = 160
    model.output_dim = 3  
    model.fc1 = torch.nn.Linear(160, 16)
    model.fc2 = torch.nn.Linear(16, 3)
    
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    except RuntimeError:
        print("Error: Model mismatch. Retrain please.")
        return

    model.eval()

    usd_balance = INITIAL_BALANCE
    btc_balance = 0.0
    portfolio_history = []
    price_history = []
    
    mid_price_history = []
    bids = {} 
    asks = {} 
    
    prev_bid_p = 0; prev_bid_v = 0
    prev_ask_p = 0; prev_ask_v = 0
    
    print(f"--- STARTING SIMULATION (Threshold: {CONFIDENCE_THRESHOLD}) ---")
    
    with h5py.File(H5_FILE, 'r') as f:
        raw_json = f['raw_json'][:]
        
        for i in tqdm(range(len(raw_json))):
            try:
                line = raw_json[i].decode('utf-8')
                data = json.loads(line)
                if not isinstance(data, list): continue
                payload = data[1]

                if 'bs' in payload or 'as' in payload:
                    
                    # SNAPSHOT
                    bids = {float(x[0]): float(x[1]) for x in payload.get('bs', [])}
                    asks = {float(x[0]): float(x[1]) for x in payload.get('as', [])}
                elif 'b' in payload or 'a' in payload:
                    
                    # UPDATE
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

                # Need valid book to proceed
                if len(bids) < 10 or len(asks) < 10: continue

                sorted_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:10]
                sorted_asks = sorted(asks.items(), key=lambda x: x[0])[:10]
                
                best_bid_p = sorted_bids[0][0]; best_bid_v = sorted_bids[0][1]
                best_ask_p = sorted_asks[0][0]; best_ask_v = sorted_asks[0][1]
                mid_price = (best_bid_p + best_ask_p) / 2
                
                mid_price_history.append(mid_price)
                if len(mid_price_history) > 10: mid_price_history.pop(0)
                if len(mid_price_history) < 10: continue 

                # OFI Calculation
                if best_bid_p > prev_bid_p: ofi_bid = best_bid_v
                elif best_bid_p < prev_bid_p: ofi_bid = -prev_bid_v
                else: ofi_bid = best_bid_v - prev_bid_v
                
                if best_ask_p > prev_ask_p: ofi_ask = prev_ask_v
                elif best_ask_p < prev_ask_p: ofi_ask = -prev_ask_v
                else: ofi_ask = best_ask_v - prev_ask_v

                net_ofi = ofi_bid - ofi_ask
                prev_bid_p = best_bid_p; prev_bid_v = best_bid_v
                prev_ask_p = best_ask_p; prev_ask_v = best_ask_v

                # Features
                volatility = np.std(mid_price_history)
                momentum = mid_price - mid_price_history[0]
                spread = best_ask_p - best_bid_p
                imbalance = (sum(v for p, v in sorted_bids) - sum(v for p, v in sorted_asks)) / (sum(v for p, v in sorted_bids) + sum(v for p, v in sorted_asks) + 1e-5)
                ofi_norm = np.tanh(net_ofi)

                feature_vector = []
                for p, v in sorted_bids:
                    norm_p = (p - mid_price) / mid_price * 1000
                    log_v = np.log(v + 1.0)
                    feature_vector.extend([norm_p, log_v, -1.0, imbalance, spread, momentum, volatility, ofi_norm])
                for p, v in sorted_asks:
                    norm_p = (p - mid_price) / mid_price * 1000
                    log_v = np.log(v + 1.0)
                    feature_vector.extend([norm_p, log_v, 1.0, imbalance, spread, momentum, volatility, ofi_norm])

                # Inference
                x = torch.tensor([feature_vector], dtype=torch.float).to(device)
                with torch.no_grad():
                    out = model(MiniBatch(x))
                    probs = torch.exp(out) 

                p_down = probs[0][0].item()
                p_flat = probs[0][1].item()
                p_up   = probs[0][2].item()

                if i < 50: 
                    print(f"Tick {i}: DOWN={p_down:.2f} | FLAT={p_flat:.2f} | UP={p_up:.2f}")

                if p_up > CONFIDENCE_THRESHOLD and usd_balance > 10:
                    btc_bought = (usd_balance / best_bid_p) * (1 - MAKER_FEE)
                    btc_balance += btc_bought
                    usd_balance = 0
                    print(f"[{i}] BUY  @ {best_bid_p:.2f} (Prob: {p_up:.2f})")
                    
                elif p_down > CONFIDENCE_THRESHOLD and btc_balance > 0.0001:
                    usd_received = (btc_balance * best_ask_p) * (1 - MAKER_FEE)
                    usd_balance += usd_received
                    btc_balance = 0
                    print(f"[{i}] SELL @ {best_ask_p:.2f} (Prob: {p_down:.2f})")

                current_val = usd_balance + (btc_balance * mid_price)
                portfolio_history.append(current_val)
                price_history.append(mid_price)

            except Exception as e:
                # print(e)
                continue

    if len(portfolio_history) > 0:
        final_value = portfolio_history[-1]
        profit_pct = ((final_value - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        
        print("\n" + "="*40)
        print(f" FINAL BALANCE: ${final_value:.2f}")
        print(f" PROFIT:        {profit_pct:.2f}%")
        print("="*40 + "\n")
        
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = 'tab:blue'
        ax1.plot(portfolio_history, color=color, label='AI Portfolio')
        ax2 = ax1.twinx()  
        color = 'tab:gray'
        ax2.plot(price_history, color=color, alpha=0.3, label='BTC Price')
        plt.show()

if __name__ == "__main__":
    backtest()