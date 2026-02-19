import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LOBDataset
import os

INPUT_DIM = 160
HIDDEN1_DIM = 64
HIDDEN2_DIM = 32
OUTPUT_DIM = 3

class MarketDN(nn.Module):
    def __init__(self):
        super(MarketDN, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1_DIM)
        self.fc2 = nn.Linear(HIDDEN1_DIM, HIDDEN2_DIM)
        self.fc3 = nn.Linear(HIDDEN2_DIM, OUTPUT_DIM)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    dataset = LOBDataset("../data/kraken_volatility.h5")
    
    y_all = torch.stack([y for _, y in dataset])
    counts = torch.bincount(y_all)
    print(f"VERIFICATION COUNTS: {counts}") 
    
    # Split
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = MarketDN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("--- STARTING TRAINING ---")
    for epoch in range(10): 
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = F.nll_loss(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == y_batch).sum())
            total_samples += y_batch.size(0)
            
        train_acc = correct / total_samples
        train_loss = total_loss / len(train_loader)

        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                test_loss += F.nll_loss(out, y_batch).item()
                pred = out.argmax(dim=1)
                test_correct += int((pred == y_batch).sum())
                test_total += y_batch.size(0)

        test_acc = test_correct / test_total
        test_loss /= len(test_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Test Loss: {test_loss:.4f} Acc: {test_acc*100:.2f}%")
        
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/market_mlp.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()