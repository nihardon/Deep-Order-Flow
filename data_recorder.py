import asyncio
import websockets
import json
import h5py
import datetime
import numpy as np

# CONFIGURATION
PAIR = "XBT/USD"
URI = "wss://ws.kraken.com"

BUFFER_SIZE = 1000 
OUTPUT_FILE = "kraken_lob_data.h5"

async def record_order_book():
    async with websockets.connect(URI) as websocket:
        print(f"Connected to Kraken Feed for {PAIR}...")

        # Subscribe to the book channel
        subscribe_message = {
            "event": "subscribe",
            "pair": [PAIR],
            "subscription": {
                "name": "book",
                "depth": 10
            }
        }
        await websocket.send(json.dumps(subscribe_message))

        # Buffers
        buffer_timestamps = []
        buffer_types = []  
        buffer_data = []
        
        msg_count = 0

        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                # Kraken sends data as a list: [CHANNEL_ID, DATA, CHANNEL_NAME, PAIR]
                if isinstance(data, list):
                    
                    # Store in RAM
                    current_time = datetime.datetime.now().isoformat().encode('utf-8')
                    buffer_timestamps.append(current_time)
                    
                    # Label it 'snapshot' or 'update' based on keys
                    # 'bs' = bid snapshot, 'as' = ask snapshot (Initial)
                    # 'b' = bid update, 'a' = ask update (Changes)
                    payload = data[1]
                    msg_type = "unknown"
                    if "as" in payload or "bs" in payload:
                        msg_type = "snapshot"
                    elif "a" in payload or "b" in payload:
                        msg_type = "update"
                    
                    buffer_types.append(msg_type.encode('utf-8'))
                    buffer_data.append(message.encode('utf-8'))
                    
                    msg_count += 1

                    # Flush to disk
                    if len(buffer_data) >= BUFFER_SIZE:
                        save_to_hdf5(buffer_timestamps, buffer_types, buffer_data)
                        print(f"Saved chunk to {OUTPUT_FILE}. Total messages: {msg_count}")
                        
                        buffer_timestamps = []
                        buffer_types = []
                        buffer_data = []

            except Exception as e:
                print(f"Error: {e}")
                break

def save_to_hdf5(timestamps, types, data):
    with h5py.File(OUTPUT_FILE, 'a') as f:
        if 'timestamp' not in f:
            dt_str = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('timestamp', data=timestamps, maxshape=(None,), chunks=True, dtype=dt_str)
            f.create_dataset('type', data=types, maxshape=(None,), chunks=True, dtype=dt_str)
            f.create_dataset('raw_json', data=data, maxshape=(None,), chunks=True, dtype=dt_str)
        else:
            new_size = f['timestamp'].shape[0] + len(timestamps)
            f['timestamp'].resize((new_size,))
            f['type'].resize((new_size,))
            f['raw_json'].resize((new_size,))
            f['timestamp'][-len(timestamps):] = timestamps
            f['type'][-len(types):] = types
            f['raw_json'][-len(data):] = data

if __name__ == "__main__":
    try:
        asyncio.run(record_order_book())
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")