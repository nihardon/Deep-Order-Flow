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
        
        print("--- REPLAYING MARKET DATA TO BUILD GRAPHS ---")
        self.process_file()
            
    def process_file(self):
        """
        Reads the HDF5 file and robustly handles variable-length Kraken messages.
        """
        self.bids = {} 
        self.asks = {}
        
        with h5py.File(self.h5_file, 'r') as f:
            timestamps = f['timestamp'][:]
            raw_json = f['raw_json'][:]
            
            total_msgs = len(timestamps) - self.lookahead
            
            for i in tqdm(range(total_msgs), desc="Building Dataset"):
                try:
                    msg_json = json.loads(raw_json[i].decode('utf-8'))
                    
                    # Handle Kraken Format
                    if isinstance(msg_json, list):
                        data = msg_json[1]
                        
                        # Case: Snapshot
                        if 'bs' in data or 'as' in data:
                            # Use item[0] (Price) and item[1] (Volume) regardless of length
                            self.bids = {float(x[0]): float(x[1]) for x in data.get('bs', [])}
                            self.asks = {float(x[0]): float(x[1]) for x in data.get('as', [])}
                            
                        # Case: Update
                        elif 'b' in data or 'a' in data:
                            # Update Bids
                            for item in data.get('b', []):
                                p = float(item[0])
                                v = float(item[1])
                                if v == 0.0: 
                                    self.bids.pop(p, None) 
                                else: 
                                    self.bids[p] = v
                            
                            # Update Asks
                            for item in data.get('a', []):
                                p = float(item[0])
                                v = float(item[1])
                                if v == 0.0: 
                                    self.asks.pop(p, None)
                                else: 
                                    self.asks[p] = v
                    
                    # --- SNAPSHOT TAKEN ---
                    if len(self.bids) >= 10 and len(self.asks) >= 10:
                        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:10]
                        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:10]
                        
                        mid_price = (sorted_bids[0][0] + sorted_asks[0][0]) / 2
                        self.samples.append((i, sorted_bids, sorted_asks, mid_price))

                except Exception as e:
                    continue

    def len(self):
        return len(self.samples)

    def get(self, idx):
        current_index, bids, asks, current_mid = self.samples[idx]
        
        # Node Features [Price, Vol, Side]
        node_features = []
        
        # Bids
        for p, v in bids:
            norm_p = (p - current_mid) / current_mid * 1000 
            node_features.append([norm_p, v, -1.0])
            
        # Asks
        for p, v in asks:
            norm_p = (p - current_mid) / current_mid * 1000
            node_features.append([norm_p, v, 1.0])
            
        x = torch.tensor(node_features, dtype=torch.float)

        # Edges
        edge_index = []
        for i in range(9):
            edge_index.append([i, i+1]); edge_index.append([i+1, i])
        for i in range(10, 19):
            edge_index.append([i, i+1]); edge_index.append([i+1, i])
        edge_index.append([0, 10]); edge_index.append([10, 0])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Label Generation
        future_idx = min(idx + 10, len(self.samples) - 1)
        future_mid = self.samples[future_idx][3]
        
        threshold = 0.00005 
        pct_change = (future_mid - current_mid) / current_mid
        
        if pct_change > threshold:
            y = torch.tensor([2], dtype=torch.long) # UP
        elif pct_change < -threshold:
            y = torch.tensor([0], dtype=torch.long) # DOWN
        else:
            y = torch.tensor([1], dtype=torch.long) # FLAT

        return Data(x=x, edge_index=edge_index, y=y)