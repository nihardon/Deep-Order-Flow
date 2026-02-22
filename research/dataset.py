import h5py
import torch
from torch.utils.data import Dataset
import json
import numpy as np
import random
from tqdm import tqdm

class LOBDataset(Dataset):
    def __init__(self, h5_file, lookahead=15): 
        """
        Args:
            h5_file (str): Path to the .h5 file containing 'raw_json' dataset.
            lookahead (int): How many ticks into the future to predict price movement.
        """
        super().__init__()
        self.h5_file = h5_file
        self.lookahead = lookahead
        self.samples = [] 
        
        print(f"--- PROCESSING DATASET ({h5_file}) ---")
        self.process_file()
            
    def process_file(self):
        up_samples = []
        down_samples = []
        flat_samples = []
        
        # Order Book State (Price -> Volume)
        bids = {} 
        asks = {}
        
        # History Tracking
        mid_prices = []
        prev_bid_p = 0; prev_bid_v = 0
        prev_ask_p = 0; prev_ask_v = 0
        
        buffer = [] 
        
        with h5py.File(self.h5_file, 'r') as f:
            if 'raw_json' not in f:
                print("Error: 'raw_json' dataset not found in file.")
                return

            raw_data = f['raw_json'][:]
            total_msgs = len(raw_data)
            
            print(f"Scanning {total_msgs} market events...")

            for i in tqdm(range(total_msgs), desc="Replaying Market"):
                try:
                    line = raw_data[i]
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    
                    data = json.loads(line)
                    
                    if not isinstance(data, list) or len(data) < 2: 
                        continue
                        
                    payload = data[1]
                    if not isinstance(payload, dict): 
                        continue

                    # Update Order Book State

                    # Update Bids
                    for item in payload.get('bs', []) + payload.get('b', []):
                        price = float(item[0])
                        volume = float(item[1])
                        if volume == 0:
                            bids.pop(price, None) 
                        else:
                            bids[price] = volume  
                            
                    # Update Asks
                    for item in payload.get('as', []) + payload.get('a', []):
                        price = float(item[0])
                        volume = float(item[1])
                        if volume == 0:
                            asks.pop(price, None)
                        else:
                            asks[price] = volume

                    if len(bids) < 10 or len(asks) < 10: 
                        continue

                    # Sort & Extract Best Prices
                    
                    # Bids = Descending (Highest buy is best)
                    sorted_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:10]
                    
                    # Asks = Ascending (Lowest sell is best)
                    sorted_asks = sorted(asks.items(), key=lambda x: x[0])[:10]
                    
                    best_bid_p = sorted_bids[0][0]; best_bid_v = sorted_bids[0][1]
                    best_ask_p = sorted_asks[0][0]; best_ask_v = sorted_asks[0][1]
                    mid_price = (best_bid_p + best_ask_p) / 2
                    mid_prices.append(mid_price)
                    
                    # Calculate Order Flow Imbalance (OFI)
                    ofi_bid = 0
                    if best_bid_p > prev_bid_p: ofi_bid = best_bid_v
                    elif best_bid_p < prev_bid_p: ofi_bid = -prev_bid_v
                    else: ofi_bid = best_bid_v - prev_bid_v
                    
                    ofi_ask = 0
                    if best_ask_p > prev_ask_p: ofi_ask = prev_ask_v
                    elif best_ask_p < prev_ask_p: ofi_ask = -prev_ask_v
                    else: ofi_ask = best_ask_v - prev_ask_v
                    
                    net_ofi = ofi_bid - ofi_ask
                    
                    # Update previous state
                    prev_bid_p = best_bid_p; prev_bid_v = best_bid_v
                    prev_ask_p = best_ask_p; prev_ask_v = best_ask_v

                    # Create Sample
                    if len(mid_prices) > 10:
                        recent_history = mid_prices[-10:]
                        sample_data = {
                            'mid': mid_price,
                            'hist': recent_history,
                            'bids': sorted_bids,
                            'asks': sorted_asks,
                            'ofi': net_ofi
                        }
                        buffer.append(sample_data)
                        
                        # Labeling
                        if len(buffer) > self.lookahead:
                            old_sample = buffer.pop(0) 
                            start_price = old_sample['mid']
                            end_price = mid_price 
                            
                            # Percentage return
                            # (End - Start) / Start
                            ret = (end_price - start_price) / start_price
                            
                            threshold = 0.00005 
                            
                            if ret > threshold:
                                up_samples.append(self.create_sample(old_sample, 2))
                            elif ret < -threshold:
                                down_samples.append(self.create_sample(old_sample, 0))
                            else:
                                if random.random() < 0.20: 
                                    flat_samples.append(self.create_sample(old_sample, 1))

                except Exception:
                    continue
        
        # Balance the Dataset
        print(f"DEBUG: Found {len(up_samples)} UP, {len(down_samples)} DOWN, {len(flat_samples)} FLAT")
        
        # Robust Minimum Size
        if len(up_samples) == 0 or len(down_samples) == 0:
            print("WARNING: Dataset is missing a class (likely DOWN). Training will be imbalanced.")
            min_len = min(len(flat_samples), max(len(up_samples), len(down_samples)))
            if min_len == 0: min_len = 100 
        else:
            min_len = min(len(up_samples), len(down_samples))

        # Truncate to balance
        up_samples = up_samples[:min_len]
        down_samples = down_samples[:min_len]
        flat_samples = flat_samples[:min_len]

        self.samples = up_samples + down_samples + flat_samples
        random.shuffle(self.samples)
        
        print(f"FINAL DATASET: {len(self.samples)} balanced samples.")
        print(f"   - UP:   {len(up_samples)}")
        print(f"   - DOWN: {len(down_samples)}")
        print(f"   - FLAT: {len(flat_samples)}")

    def create_sample(self, sample, label):
        """
        Converts raw market data into a flat feature vector + label.
        Features are normalized to be Percentage/Relative.
        """
        current_mid = sample['mid']
        history = sample['hist']
        bids = sample['bids']
        asks = sample['asks']
        ofi = sample['ofi']
                
        # Volatility (Std Dev of % Returns of the last 10 ticks)
        hist_pct = [(history[i] - history[i-1])/history[i-1] for i in range(1, len(history))]
        if len(hist_pct) == 0: hist_pct = [0.0]
        volatility = np.std(hist_pct) * 1000
        
        # Momentum (% Change from 10 ticks ago)
        momentum = (current_mid - history[0]) / history[0] * 1000 
        
        # Spread (% of price)
        spread = (asks[0][0] - bids[0][0]) / current_mid * 1000
        
        # OFI (Normalized)
        ofi_norm = np.tanh(ofi) 
        
        # Volume Imbalance
        total_bid_vol = sum(v for p, v in bids)
        total_ask_vol = sum(v for p, v in asks)
        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-5)

        # [PriceDist, LogVol, Side, Imbalance, Spread, Momentum, Volatility, OFI] per level
        features = []
        
        for p, v in bids:
            norm_p = (p - current_mid) / current_mid * 1000 
            log_v = np.log(v + 1.0) 
            features.extend([norm_p, log_v, -1.0, imbalance, spread, momentum, volatility, ofi_norm])
            
        for p, v in asks:
            norm_p = (p - current_mid) / current_mid * 1000
            log_v = np.log(v + 1.0)
            features.extend([norm_p, log_v, 1.0, imbalance, spread, momentum, volatility, ofi_norm])

        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.long)
        return (x, y)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]