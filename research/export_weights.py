import torch
import struct
from train import MarketDN

model = MarketDN()
model.load_state_dict(torch.load("../models/market_mlp.pth", map_location='cpu'))
model.eval()

print("Exporting weights to ../data/model_weights.bin ...")

def write_layer(f, layer):
    w = layer.weight.detach().numpy().flatten()
    b = layer.bias.detach().numpy().flatten()
    f.write(struct.pack(f'{len(w)}f', *w))
    f.write(struct.pack(f'{len(b)}f', *b))

with open("../data/model_weights.bin", "wb") as f:
    write_layer(f, model.fc1)  # 64 x 160 + 64
    write_layer(f, model.fc2)  # 32 x 64  + 32
    write_layer(f, model.fc3)  # 3  x 32  + 3

print("Done. C++ Engine is ready to load.")