# Deep Order Flow

A real-time BTC/USD scalping system that uses a neural network trained on limit order book (LOB) microstructure features to predict short-term price movements. The model runs inference in C++ for low-latency execution while Python handles data collection, training, and live monitoring.

## Architecture

```
┌─────────────┐       binary stdin       ┌─────────────┐       text stdout       ┌─────────────┐
│  live_feed  │ ──────────────────────►  │  hft_engine │ ──────────────────────► │ dashboard   │
│  (Python)   │   160 features + price   │  (C++17)    │   TICK / ACTION msgs    │ (Python)    │
│             │                          │             │                         │             │
│  WebSocket  │                          │  MLP Infer  │                         │ Paper Trade │
│  Order Book │                          │  Buy/Sell   │                         │ Win Rate    │
│  Features   │                          │  Logic      │                         │ PnL         │
└─────────────┘                          └─────────────┘                         └─────────────┘
```

The pipeline is connected via Unix pipes: `live_feed.py | ./hft_engine | dashboard.py`

## Features

Each order book snapshot produces a 160-dimensional feature vector (20 price levels x 8 features):

| Feature | Description |
|---------|-------------|
| PriceDist | Distance from mid-price (normalized, per mille) |
| LogVol | Log-scaled volume at the level |
| Side | Bid (-1) or Ask (+1) |
| Imbalance | Volume imbalance across all levels |
| Spread | Bid-ask spread (per mille) |
| Momentum | Price change over recent history (per mille) |
| Volatility | Std deviation of recent returns (per mille) |
| OFI | Order Flow Imbalance (tanh-normalized) |

## Model

A 3-layer MLP trained in PyTorch, exported to raw binary weights for C++ inference:

- **Input**: 160 (20 levels x 8 features)
- **Hidden 1**: 64 neurons, ReLU, 20% dropout
- **Hidden 2**: 32 neurons, ReLU, 20% dropout
- **Output**: 3 classes (Down / Flat / Up) via softmax

Training uses balanced class sampling with a configurable lookahead window for labeling.

## Trading Logic (C++ Engine)

The engine reads model predictions and executes a rule-based scalping strategy:

- **Entry**: Buy when P(Up) exceeds confidence threshold
- **Take Profit**: Sell when unrealized PnL hits target (default 0.05%)
- **Stop Loss**: Sell when unrealized PnL exceeds loss limit (default 0.20%)
- **AI Exit**: Sell when model predicts Down with high confidence for sustained ticks
- **Time Stop**: Force-sell positions held too long at a loss
- **Cooldowns**: Penalty cooldown after losses, shorter cooldown after wins

## Project Structure

```
├── src/
│   ├── main.cpp              # C++ inference engine + trading logic
│   └── CMakeLists.txt         # Build config (C++17, Apple Accelerate)
├── research/
│   ├── data_recorder.py       # WebSocket recorder → HDF5
│   ├── dataset.py             # LOB replay, feature engineering, labeling
│   ├── train.py               # Model training with train/test evaluation
│   ├── export_weights.py      # PyTorch → binary weight export for C++
│   ├── live_feed.py           # Real-time feature extraction → binary stdout
│   ├── dashboard.py           # Live paper trading terminal
│   └── inspect_data.py        # HDF5 data inspection utility
├── data/                      # HDF5 market data + exported weights (gitignored)
├── models/                    # Saved PyTorch models (gitignored)
└── build/                     # CMake build directory (gitignored)
```

## Getting Started

### Prerequisites

- Python 3.10+
- CMake 3.14+
- C++17 compiler

### Setup

```bash
# Python dependencies
python -m venv venv
source venv/bin/activate
pip install torch numpy websockets h5py tqdm

# Build C++ engine
mkdir build && cd build
cmake ../src
make
```

### Record Data

```bash
cd research
python data_recorder.py
# Let it run for several hours to collect order book data
```

### Train

```bash
cd research
python train.py
python export_weights.py
```

### Run Live

```bash
cd build
python3 ../research/live_feed.py | ./hft_engine | python3 ../research/dashboard.py
```

## Data Source

Live order book data from the [Kraken WebSocket API](https://docs.kraken.com/websockets/) (XBT/USD, depth 10).
