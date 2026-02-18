import h5py
import json

INPUT_FILE = "kraken_lob_data.h5"

def inspect_file():
    print(f"--- OPENING {INPUT_FILE} ---")
    
    try:
        with h5py.File(INPUT_FILE, 'r') as f:
            # Check what datasets we have
            print(f"Datasets found: {list(f.keys())}")
            
            # Get the data columns
            timestamps = f['timestamp'][:]
            types = f['type'][:]
            raw_json = f['raw_json'][:]
            
            print(f"Total Messages Captured: {len(timestamps)}\n")

            # Print the first 5 messages to understand the format
            print("--- SAMPLE DATA (First 5 Messages) ---")
            for i in range(min(5, len(timestamps))):
                t = timestamps[i].decode('utf-8')
                msg_type = types[i].decode('utf-8')
                data_str = raw_json[i].decode('utf-8')
                data_dict = json.loads(data_str)
                
                print(f"Msg #{i+1} | Time: {t} | Type: {msg_type.upper()}")
                if isinstance(data_dict, list):
                    print(f"Payload Keys: {data_dict[1].keys() if len(data_dict) > 1 else 'N/A'}")
                else:
                    print(f"Data: {data_dict}")
                print("-" * 50)
                
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Make sure it is in this folder.")

if __name__ == "__main__":
    inspect_file()