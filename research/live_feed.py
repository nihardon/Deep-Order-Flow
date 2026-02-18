import asyncio
import websockets
import json
import numpy as np
import sys
import struct


PAIR = "XBT/USD"
URI = "wss://ws.kraken.com"

mid_price_history = []
MAX_HISTORY = 10

def calculate_features(bids, asks):
    """
    Converts raw Bids/Asks into the 140-float feature vector.
    """
    global mid_price_history
    
    # Sort and Filter
    sorted_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:10]
    sorted_asks = sorted(asks.items(), key=lambda x: x[0])[:10]
    
    if len(sorted_bids) < 10 or len(sorted_asks) < 10:
        return None

    # Update History
    best_bid = sorted_bids[0][0]
    best_ask = sorted_asks[0][0]
    mid_price = (best_bid + best_ask) / 2
    
    mid_price_history.append(mid_price)
    if len(mid_price_history) > MAX_HISTORY:
        mid_price_history.pop(0)
    
    if len(mid_price_history) < MAX_HISTORY:
        return None 

    # Calculate Global Features
    volatility = np.std(mid_price_history)
    momentum = mid_price - mid_price_history[0]
    spread = best_ask - best_bid
    
    total_bid_vol = sum(v for p, v in sorted_bids)
    total_ask_vol = sum(v for p, v in sorted_asks)
    imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-5)

    feature_vector = []
    
    # Bids
    for p, v in sorted_bids:
        norm_p = (p - mid_price) / mid_price * 1000
        log_v = np.log(v + 1.0)
        feature_vector.extend([norm_p, log_v, -1.0, imbalance, spread, momentum, volatility])
        
    # Asks
    for p, v in sorted_asks:
        norm_p = (p - mid_price) / mid_price * 1000
        log_v = np.log(v + 1.0)
        feature_vector.extend([norm_p, log_v, 1.0, imbalance, spread, momentum, volatility])
        
    return feature_vector

async def stream_market_data():
    async with websockets.connect(URI) as websocket:
        # Subscribe
        msg = {
            "event": "subscribe",
            "pair": [PAIR],
            "subscription": {"name": "book", "depth": 10}
        }
        await websocket.send(json.dumps(msg))
        
        # Local Order Book
        bids = {}
        asks = {}
        
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                if isinstance(data, list):
                    payload = data[1]
                    
                    # Update Local Book (Snapshot)
                    if 'bs' in payload or 'as' in payload: 
                        bids = {float(x[0]): float(x[1]) for x in payload.get('bs', [])}
                        asks = {float(x[0]): float(x[1]) for x in payload.get('as', [])}
                    
                    # Update Local Book (Updates)
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

                    # Generate Features
                    features = calculate_features(bids, asks)
                    
                    if features:
                        binary_data = struct.pack('140f', *features)
                        sys.stdout.buffer.write(binary_data)
                        sys.stdout.buffer.flush()
                        
            except Exception as e:
                sys.stderr.write(f"Error: {e}\n")
                continue

if __name__ == "__main__":
    try:
        asyncio.run(stream_market_data())
    except KeyboardInterrupt:
        pass