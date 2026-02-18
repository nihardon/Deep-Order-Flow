import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dataset import LOBDataset
import os

class MarketDN(torch.nn.Module):
    def __init__(self):
        super(MarketDN, self).__init__()
        
        self.input_dim = 140 
        self.hidden_dim = 16
        self.output_dim = 3
        
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        # Flatten the graph into a single vector
        # [Batch, Node, Feature] -> [Batch, Node*Feature]
        x = data.x.view(data.num_graphs, -1)
        
        # Linear Layer 1
        x = self.fc1(x)
        x = F.relu(x) 
        
        # Linear Layer 2
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# The Training Loop
def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    dataset = LOBDataset("../data/kraken_lob_data.h5")
    
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = MarketDN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("--- STARTING TRAINING (DENSE MODE) ---")
    model.train()
    for epoch in range(10): 
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total_samples += batch.y.size(0)
            
        acc = correct / total_samples
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f} | Acc: {acc*100:.2f}%")
        
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/market_gnn.pth") # Keep filename same for ease
    print("Model saved to ../models/market_gnn.pth")

if __name__ == "__main__":
    train()