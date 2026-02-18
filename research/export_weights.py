import torch
import struct
import os
from train import MarketGNN

def export_model():
    model_path = "../models/market_gnn.pth"
    output_path = "../data/model_weights.bin"

    print(f"--- LOADING MODEL FROM {model_path} ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = MarketGNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with open(output_path, "wb") as f:
        print(f"--- EXPORTING WEIGHTS TO {output_path} ---")
        
        total_params = 0
        
        # This iterates through every weight/bias in the exact order PyTorch sees them.
        for name, tensor in model.named_parameters():
            # Flatten to 1D array
            data = tensor.detach().numpy().flatten()
            
            # Write as 32-bit floats
            binary_data = struct.pack(f'{len(data)}f', *data)
            f.write(binary_data)
            
            print(f"Exported: {name:20} | Shape: {str(list(tensor.shape)):15} | Count: {len(data)}")
            total_params += len(data)

    print("-" * 50)
    print(f"SUCCESS: Exported {total_params} floats to {output_path}")

if __name__ == "__main__":
    export_model()