import h5py
import json
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

INPUT_FILE = "kraken_lob_data.h5"

def load_first_snapshot():
    """
    Finds the first 'snapshot' message in the HDF5 file.
    """
    with h5py.File(INPUT_FILE, 'r') as f:
        types = f['type'][:]
        raw_json = f['raw_json'][:]
        
        # Search for the first message labeled 'snapshot'
        for i, msg_type in enumerate(types):
            if msg_type.decode('utf-8') == 'snapshot':
                print(f"Found Snapshot at Index {i}")
                return json.loads(raw_json[i].decode('utf-8'))
    
    raise ValueError("No snapshot found in the file!")

def build_graph(snapshot_json):
    """
    Converts a Kraken Snapshot JSON into a PyTorch Geometric Graph.
    """
    # Kraken data format: [channelID, {"as": [[price, vol], ...], "bs": [[price, vol], ...]}, ...]
    data = snapshot_json[1] # The dictionary is the second item
    
    # Get Top 10 Bids and Asks
    bids = data['bs'][:10] 
    asks = data['as'][:10] 
    
    node_features = []
    
    # Create Nodes
    # We label Bids as Class 0, Asks as Class 1
    for price, volume, timestamp in bids:
        # Feature Vector: [Price, Volume, Type(-1 for Bid)]
        node_features.append([float(price), float(volume), -1.0])
        
    for price, volume, timestamp in asks:
        # Feature Vector: [Price, Volume, Type(1 for Ask)]
        node_features.append([float(price), float(volume), 1.0])

    # Convert to PyTorch Tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create Edges
    edge_index = []
    
    # Connect every Bid to the next Bid, and every Ask to the next Ask
    # Also connect the Best Bid to the Best Ask
    
    # Connect Bids (0 to 9)
    for i in range(9):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i]) 
        
    # Connect Asks (10 to 19)
    for i in range(10, 19):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])
        
    # Connect Best Bid (0) to Best Ask (10) -> THE SPREAD
    edge_index.append([0, 10])
    edge_index.append([10, 0])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

def draw_market_graph(graph):
    """
    Draws the graph using NetworkX.
    """
    g = nx.Graph()
    
    # Add nodes
    for i in range(graph.num_nodes):
        price = graph.x[i][0].item()
        vol = graph.x[i][1].item()
        side = "Bid" if graph.x[i][2].item() == -1 else "Ask"
        g.add_node(i, price=price, vol=vol, side=side)
        
    # Add edges
    edges = graph.edge_index.t().tolist()
    g.add_edges_from(edges)
    
    # Drawing Setup
    pos = nx.spring_layout(g, seed=42) # Force-directed graph layout
    
    # Color Bids Green, Asks Red
    colors = ['green' if g.nodes[i]['side'] == 'Bid' else 'red' for i in g.nodes]
    sizes = [g.nodes[i]['vol'] * 1000 for i in g.nodes] # Volume determines size
    
    plt.figure(figsize=(10, 6))
    nx.draw(g, pos, with_labels=False, node_color=colors, node_size=sizes, edge_color="gray")
    
    # Add labels (Price)
    labels = {i: f"{g.nodes[i]['price']:.2f}" for i in g.nodes}
    nx.draw_networkx_labels(g, pos, labels, font_size=8)
    
    plt.title("Visualizing the Limit Order Book as a Graph")
    print("Graph generated! Check the popup window.")
    plt.show()

if __name__ == "__main__":
    snapshot = load_first_snapshot()
    graph = build_graph(snapshot)
    print("Graph Object Created:", graph)
    draw_market_graph(graph)