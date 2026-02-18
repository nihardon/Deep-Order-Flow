import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dataset import LOBDataset
import os
import numpy as np

class MarketDN(torch.nn.Module):
    def __init__(self):
        super(MarketDN, self).__init__()
        self.input_dim = 140 
        self.hidden_dim = 16
        self.output_dim = 3
        
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x = data.x.view(data.num_graphs, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    dataset = LOBDataset("../data/kraken_lob_data.h5")

    print("Calculating class weights...")
    y_all = torch.cat([data.y for data in dataset])
    
    class_counts = torch.bincount(y_all, minlength=3)
    total_samples = len(y_all)
    
    print(f"Counts - DOWN: {class_counts[0]}, HOLD: {class_counts[1]}, UP: {class_counts[2]}")
    
    class_weights = total_samples / (3 * (class_counts.float() + 1e-5))
    
    class_weights[class_counts == 0] = 0.0
    
    class_weights = class_weights.to(device)
    print(f"Weights: {class_weights}")
    print("--------------------------------")

    # Split Data
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = MarketDN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("--- STARTING TRAINING (WEIGHTED) ---")
    model.train()
    for epoch in range(10): 
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            
            loss = F.nll_loss(out, batch.y, weight=class_weights)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total_samples += batch.y.size(0)
            
        acc = correct / total_samples
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f} | Acc: {acc*100:.2f}%")
        
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/market_gnn.pth")
    print("Model saved to ../models/market_gnn.pth")

if __name__ == "__main__":
    train()