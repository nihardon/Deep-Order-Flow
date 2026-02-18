import asyncio
import websockets
import json
import h5py
import datetime
import os


PAIR = "XBT/USD"
URI = "wss://ws.kraken.com"
BUFFER_SIZE = 1000 

OUTPUT_FILE = "../data/kraken_volatility.h5" 

async def record_order_book():
    print(f"--- FAST RECORDER (WebSockets) ---")
    print(f"Target: {OUTPUT_FILE}")
    print(f"Connecting to {URI}...")
    
    async with websockets.connect(URI) as websocket:
        print(f"Connected! Subscribing to {PAIR}...")

        # Subscribe to order book
        subscribe_message = {
            "event": "subscribe",
            "pair": [PAIR],
            "subscription": {
                "name": "book",
                "depth": 10
            }
        }
        await websocket.send(json.dumps(subscribe_message))

        buffer_data = []
        msg_count = 0

        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                # Kraken sends updates as lists: [channelID, payload, ...]
                if isinstance(data, list):
                    
                    buffer_data.append(message)
                    msg_count += 1

                    # Visual Heartbeat (Show speed)
                    if msg_count % 100 == 0:
                        print(f"Captured: {msg_count} updates...", end='\r')

                    # Flush to Disk
                    if len(buffer_data) >= BUFFER_SIZE:
                        save_to_hdf5(buffer_data)
                        buffer_data = [] # Clear memory

            except Exception as e:
                print(f"Error: {e}")
                break

def save_to_hdf5(data_list):
    # Ensure directory exists
    os.makedirs("../data", exist_ok=True)
    
    with h5py.File(OUTPUT_FILE, 'a') as f:
        # Create dataset if it doesn't exist
        if 'raw_json' not in f:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('raw_json', (0,), maxshape=(None,), dtype=dt)
        
        dset = f['raw_json']
        
        # Resize and Append
        current_len = dset.shape[0]
        new_len = current_len + len(data_list)
        dset.resize((new_len,))
        
        dset[current_len:] = data_list

if __name__ == "__main__":
    try:
        asyncio.run(record_order_book())
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")