import torch
import struct
import os
from train import MarketDN

def export_model():
    model_path = "../models/market_gnn.pth"
    output_path = "../data/model_weights.bin"

    print(f"--- LOADING MODEL FROM {model_path} ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = MarketDN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with open(output_path, "wb") as f:
        print(f"--- EXPORTING WEIGHTS TO {output_path} ---")
        total_params = 0
        
        # Export parameters in order: fc1.weight, fc1.bias, fc2.weight, fc2.bias
        for name, tensor in model.named_parameters():
            data = tensor.detach().numpy().flatten()
            binary_data = struct.pack(f'{len(data)}f', *data)
            f.write(binary_data)
            
            print(f"Exported: {name:20} | Shape: {str(list(tensor.shape)):15} | Count: {len(data)}")
            total_params += len(data)

    print("-" * 50)
    print(f"SUCCESS: Exported {total_params} floats to {output_path}")

if __name__ == "__main__":
    export_model()