import asyncio
import websockets
import json
import h5py
import datetime
import os

# CONFIGURATION
PAIR = "XBT/USD"
URI = "wss://ws.kraken.com"
BUFFER_SIZE = 1000 
OUTPUT_FILE = "../data/kraken_volatility.h5" 

async def connect_and_record():
    while True:
        try:
            print(f"Connecting to {URI}...")
            async with websockets.connect(URI) as websocket:
                print(f"Connected! Subscribing to {PAIR}...")

                subscribe_message = {
                    "event": "subscribe",
                    "pair": [PAIR],
                    "subscription": {"name": "book", "depth": 10}
                }
                await websocket.send(json.dumps(subscribe_message))

                buffer_data = []
                msg_count = 0

                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if isinstance(data, list):
                        buffer_data.append(message)
                        msg_count += 1

                        if msg_count % 100 == 0:
                            print(f"Captured: {msg_count} updates...", end='\r')

                        if len(buffer_data) >= BUFFER_SIZE:
                            save_to_hdf5(buffer_data)
                            buffer_data = [] 

        except Exception as e:
            print(f"\nConnection lost: {e}")
            print("Reconnecting in 5 seconds...")
            await asyncio.sleep(5) # Wait before reconnecting

def save_to_hdf5(data_list):
    os.makedirs("../data", exist_ok=True)
    with h5py.File(OUTPUT_FILE, 'a') as f:
        if 'raw_json' not in f:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('raw_json', (0,), maxshape=(None,), dtype=dt)
        dset = f['raw_json']
        current_len = dset.shape[0]
        new_len = current_len + len(data_list)
        dset.resize((new_len,))
        dset[current_len:] = data_list

if __name__ == "__main__":
    try:
        asyncio.run(connect_and_record())
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")