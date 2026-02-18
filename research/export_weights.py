import torch
import struct
from train import MarketDN

# Load model
model = MarketDN()
model.input_dim = 160
model.output_dim = 3
model.fc1 = torch.nn.Linear(160, 16)
model.fc2 = torch.nn.Linear(16, 3)

model.load_state_dict(torch.load("../models/market_gnn.pth", map_location='cpu'))
model.eval()

print("Exporting weights to ../data/model_weights.bin ...")

with open("../data/model_weights.bin", "wb") as f:
    # Layer 1 Weights (16 x 160)
    w1 = model.fc1.weight.detach().numpy().flatten()
    f.write(struct.pack(f'{len(w1)}f', *w1))
    
    # Layer 1 Bias (16)
    b1 = model.fc1.bias.detach().numpy().flatten()
    f.write(struct.pack(f'{len(b1)}f', *b1))
    
    # Layer 2 Weights (3 x 16)
    w2 = model.fc2.weight.detach().numpy().flatten()
    f.write(struct.pack(f'{len(w2)}f', *w2))
    
    # Layer 2 Bias (3)
    b2 = model.fc2.bias.detach().numpy().flatten()
    f.write(struct.pack(f'{len(b2)}f', *b2))

print("Done. C++ Engine is ready to load.")