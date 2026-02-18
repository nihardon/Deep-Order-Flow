import h5py
import torch
import json
import numpy as np
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

class LOBDataset(Dataset):
    def __init__(self, h5_file, lookahead=20):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.h5_file = h5_file
        self.lookahead = lookahead
        self.samples = [] 
        
        print("--- REPLAYING MARKET DATA (SMART MODE) ---")
        self.process_file()
            
    def process_file(self):
        self.bids = {} 
        self.asks = {}
        
        # History for calculating volatility/momentum
        self.mid_prices = []
        
        with h5py.File(self.h5_file, 'r') as f:
            timestamps = f['timestamp'][:]
            raw_json = f['raw_json'][:]
            total_msgs = len(timestamps) - self.lookahead
            
            for i in tqdm(range(total_msgs), desc="Feature Engineering"):
                try:
                    msg_json = json.loads(raw_json[i].decode('utf-8'))
                    
                    if isinstance(msg_json, list):
                        data = msg_json[1]
                        if 'bs' in data or 'as' in data:
                            self.bids = {float(x[0]): float(x[1]) for x in data.get('bs', [])}
                            self.asks = {float(x[0]): float(x[1]) for x in data.get('as', [])}
                        elif 'b' in data or 'a' in data:
                            for item in data.get('b', []):
                                p, v = float(item[0]), float(item[1])
                                if v == 0: self.bids.pop(p, None) 
                                else: self.bids[p] = v
                            for item in data.get('a', []):
                                p, v = float(item[0]), float(item[1])
                                if v == 0: self.asks.pop(p, None)
                                else: self.asks[p] = v
                    
                    if len(self.bids) >= 10 and len(self.asks) >= 10:
                        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:10]
                        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:10]
                        
                        mid_price = (sorted_bids[0][0] + sorted_asks[0][0]) / 2
                        self.mid_prices.append(mid_price)
                        
                        if len(self.mid_prices) > 10:
                            # Save the snapshot and the history needed for momentum
                            recent_history = self.mid_prices[-10:]
                            self.samples.append((i, sorted_bids, sorted_asks, mid_price, recent_history))

                except Exception:
                    continue

    def len(self):
        return len(self.samples)

    def get(self, idx):
        # Unpack data
        current_index, bids, asks, current_mid, history = self.samples[idx]
                
        # Volatility (Std Dev of last 10 ticks)
        volatility = np.std(history)
        
        # Momentum (Price change over last 10 ticks)
        momentum = (current_mid - history[0])
        
        # Spread (The cost of trading)
        spread = asks[0][0] - bids[0][0]
        
        # Imbalance (Buying pressure vs Selling pressure)
        total_bid_vol = sum([v for p, v in bids])
        total_ask_vol = sum([v for p, v in asks])
        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-5)

        node_features = []
        
        # Build Bids
        for p, v in bids:
            norm_p = (p - current_mid) / current_mid * 1000 
            log_v = np.log(v + 1.0) # Log scale volume
            # Feature Vector: [Price, Vol, Side, Imbalance, Spread, Momentum, Volatility]
            node_features.append([norm_p, log_v, -1.0, imbalance, spread, momentum, volatility])
            
        # Build Asks
        for p, v in asks:
            norm_p = (p - current_mid) / current_mid * 1000
            log_v = np.log(v + 1.0)
            node_features.append([norm_p, log_v, 1.0, imbalance, spread, momentum, volatility])
            
        x = torch.tensor(node_features, dtype=torch.float)

        # Edges 
        edge_index = []
        for i in range(9): edge_index += [[i, i+1], [i+1, i]]
        for i in range(10, 19): edge_index += [[i, i+1], [i+1, i]]
        edge_index += [[0, 10], [10, 0]]
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Label
        future_idx = min(idx + 10, len(self.samples) - 1)
        future_mid = self.samples[future_idx][3]
        threshold = 0.00005 
        pct_change = (future_mid - current_mid) / current_mid
        
        if pct_change > threshold: y = torch.tensor([2], dtype=torch.long)
        elif pct_change < -threshold: y = torch.tensor([0], dtype=torch.long)
        else: y = torch.tensor([1], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)