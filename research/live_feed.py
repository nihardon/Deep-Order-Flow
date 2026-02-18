import ccxt
import time
import numpy as np
import struct
import sys

def get_features(bids, asks, mid_history, prev_ofi):
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / 2
    
    mid_history.append(mid_price)
    if len(mid_history) > 10: mid_history.pop(0)
    
    if len(mid_history) < 10: return None, mid_price, prev_ofi

    
    # Volatility (Std Dev of % Returns)
    hist_pct = [(mid_history[i] - mid_history[i-1])/mid_history[i-1] for i in range(1, len(mid_history))]
    if len(hist_pct) == 0: hist_pct = [0.0]
    volatility = np.std(hist_pct) * 1000
    
    # Momentum (% Change)
    momentum = (mid_price - mid_history[0]) / mid_history[0] * 1000
    
    # Spread (% of price)
    spread = (best_ask - best_bid) / mid_price * 1000
    
    # Imbalance
    bid_vol = sum(item[1] for item in bids)
    ask_vol = sum(item[1] for item in asks)
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-5)
    
    # OFI
    ofi_norm = np.tanh(imbalance)

    features = []
    
    for item in bids:
        p = item[0]; v = item[1]
        norm_p = (p - mid_price) / mid_price * 1000
        log_v = np.log(v + 1.0)
        features.extend([norm_p, log_v, -1.0, imbalance, spread, momentum, volatility, ofi_norm])
        
    for item in asks:
        p = item[0]; v = item[1]
        norm_p = (p - mid_price) / mid_price * 1000
        log_v = np.log(v + 1.0)
        features.extend([norm_p, log_v, 1.0, imbalance, spread, momentum, volatility, ofi_norm])
        
    return features, mid_price, ofi_norm

def main():
    kraken = ccxt.kraken({'enableRateLimit': False})
    symbol = 'BTC/USDT'
    mid_history = []
    prev_ofi = 0
    
    sys.stderr.write(f"--- LIVE FEED (Percentage Mode) ---\n")
    
    while True:
        try:
            orderbook = kraken.fetch_order_book(symbol, limit=10)
            bids = orderbook['bids']; asks = orderbook['asks']
            
            if len(bids) < 10 or len(asks) < 10: continue
            
            features, mid_price, prev_ofi = get_features(bids, asks, mid_history, prev_ofi)
            
            if features:
                data = features + [mid_price] 
                packed_data = struct.pack(f'{len(data)}f', *data)
                sys.stdout.buffer.write(packed_data)
                sys.stdout.flush()
                sys.stderr.write(".")
                sys.stderr.flush()
            else:
                sys.stderr.write("w"); sys.stderr.flush()
            
            time.sleep(1.0) 
            
        except Exception as e:
            sys.stderr.write(f"\n[Error] {e}\n")
            time.sleep(5) 
            continue

if __name__ == "__main__":
    main()