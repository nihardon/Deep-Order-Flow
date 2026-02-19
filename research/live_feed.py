import asyncio
import websockets
import json
import sys
import numpy as np

# CONFIGURATION
PAIR = "XBT/USD"
URI = "wss://ws.kraken.com"

HISTORY_LENGTH = 100 

async def run_feed():
    async with websockets.connect(URI) as websocket:
        subscribe_msg = {
            "event": "subscribe",
            "pair": [PAIR],
            "subscription": {"name": "book", "depth": 10}
        }
        await websocket.send(json.dumps(subscribe_msg))

        # Buffers
        history = []
        prev_bid = 0
        prev_ask = 0
        
        # Order Book State
        bids = {}
        asks = {}

        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                # Expect list [channelID, payload, ...]
                if not isinstance(data, list): continue
                
                payload = data[1]
                
                # Update Book
                updates = []
                if isinstance(payload, dict):
                    updates += payload.get('bs', []) + payload.get('b', []) # Bids
                    updates += payload.get('as', []) + payload.get('a', []) # Asks
                    

                    for item in payload.get('bs', []) + payload.get('b', []):
                        bids[float(item[0])] = float(item[1])
                    for item in payload.get('as', []) + payload.get('a', []):
                        asks[float(item[0])] = float(item[1])
                        
                    # Clean zero volume levels
                    bids = {k:v for k,v in bids.items() if v > 0}
                    asks = {k:v for k,v in asks.items() if v > 0}
                
                if len(bids) < 1 or len(asks) < 1: continue

                # Sort
                best_bid = max(bids.keys())
                best_ask = min(asks.keys())
                mid_price = (best_bid + best_ask) / 2.0
                
                # Update History
                history.append(mid_price)
                if len(history) > HISTORY_LENGTH: history.pop(0)
                
                # Need full history to start
                if len(history) < HISTORY_LENGTH: continue

                
                # Volatility
                hist_pct = [(history[i] - history[i-1])/history[i-1] for i in range(1, len(history))]
                if len(hist_pct) == 0: hist_pct = [0.0]
                volatility = np.std(hist_pct) * 1000

                # Momentum
                momentum = (mid_price - history[0]) / history[0] * 1000
                
                # Spread
                spread = (best_ask - best_bid) / mid_price * 1000
                
                # OFI
                ofi = 0
                ofi_norm = np.tanh(ofi)
                
                # Imbalance
                imbalance = 0.0
                
                # [PriceDist, LogVol, Side, Imbalance, Spread, Momentum, Volatility, OFI]                
                feature_vector = []
                
                sorted_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:10]
                sorted_asks = sorted(asks.items(), key=lambda x: x[0])[:10]
                
                # Fill Bids
                for i in range(10):
                    p, v = sorted_bids[i] if i < len(sorted_bids) else (best_bid, 0)
                    norm_p = (p - mid_price) / mid_price * 1000
                    log_v = np.log(v + 1.0)
                    feature_vector.extend([norm_p, log_v, -1.0, imbalance, spread, momentum, volatility, ofi_norm])
                    
                # Fill Asks
                for i in range(10):
                    p, v = sorted_asks[i] if i < len(sorted_asks) else (best_ask, 0)
                    norm_p = (p - mid_price) / mid_price * 1000
                    log_v = np.log(v + 1.0)
                    feature_vector.extend([norm_p, log_v, 1.0, imbalance, spread, momentum, volatility, ofi_norm])
                
                feature_vector.append(mid_price)
                
                packed_data = np.array(feature_vector, dtype=np.float32).tobytes()
                sys.stdout.buffer.write(packed_data)
                sys.stdout.buffer.flush()

            except Exception:
                continue

if __name__ == "__main__":
    try:
        asyncio.run(run_feed())
    except KeyboardInterrupt:
        pass