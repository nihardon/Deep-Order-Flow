import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from dataset import LOBDataset

# GNN Architecture
class MarketGNN(torch.nn.Module):
    def __init__(self):
        super(MarketGNN, self).__init__()
        
        # GAT Layer 1: Input 3 features (Price, Vol, Side) -> Hidden 16
        self.conv1 = GATConv(3, 16, heads=2, concat=True) 
        # Output size = 16 * 2 heads = 32
        
        # GAT Layer 2: Hidden 32 -> Hidden 16
        self.conv2 = GATConv(32, 16, heads=1, concat=False)
        
        # Fully Connected Classifier
        self.fc = torch.nn.Linear(16, 3) # Output: 3 classes (Down, Hold, Up)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Pooling (Summarize the whole graph into one vector)
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Training Loop
def train():
    # Detect M2 Chip or use CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # Load Data
    dataset = LOBDataset("../data/kraken_lob_data.h5")
    # Split: 80% Train, 20% Test
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Init model
    model = MarketGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Loop
    print("--- STARTING TRAINING ---")
    model.train()
    for epoch in range(5): # Run 5 times over the data
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = F.nll_loss(out, batch.y) # Negative Log Likelihood Loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")
        
    # Save the Brain
    torch.save(model.state_dict(), "../models/market_gnn.pth")
    print("Model saved to market_gnn.pth")

if __name__ == "__main__":
    train()