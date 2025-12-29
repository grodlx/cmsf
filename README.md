# Polymarket RL Trading

Reinforcement learning system for trading 15-minute binary prediction markets on Polymarket.

## Overview

This system uses PPO (Proximal Policy Optimization) with MLX for Apple Silicon GPU acceleration to learn trading strategies on Polymarket's 15-minute crypto prediction markets (BTC, ETH, SOL, XRP).

The markets are binary options that resolve to $1 or $0 based on whether the underlying asset's price is above or below its opening price at market close.

## Architecture

```
├── run.py                    # Main trading engine
├── dashboard.py              # Real-time Flask-SocketIO web dashboard
├── strategies/
│   ├── base.py               # Base classes (Action, MarketState, Strategy)
│   ├── rl_mlx.py             # PPO implementation with MLX
│   ├── momentum.py           # Momentum baseline
│   ├── mean_revert.py        # Mean reversion baseline
│   ├── fade_spike.py         # Spike fading baseline
│   └── gating.py             # Ensemble gating strategy
└── helpers/
    ├── polymarket_api.py     # Polymarket REST API
    ├── binance_wss.py        # Binance spot price streaming
    ├── binance_futures.py    # Futures data (funding, OI, CVD, liquidations)
    └── orderbook_wss.py      # Polymarket CLOB orderbook streaming
```

## Features

### State Space (18 dimensions)

| Category | Features | Description |
|----------|----------|-------------|
| **Momentum** | `returns_1m`, `returns_5m`, `returns_10m` | Ultra-short price momentum |
| **Order Flow** | `ob_imbalance_l1`, `ob_imbalance_l5`, `trade_flow`, `cvd_accel` | Buy/sell pressure signals |
| **Microstructure** | `spread_pct`, `trade_intensity`, `large_trade_flag` | Market quality metrics |
| **Volatility** | `vol_5m`, `vol_expansion` | Short-term vol regime |
| **Position** | `has_position`, `position_side`, `position_pnl`, `time_remaining` | Current exposure |
| **Regime** | `vol_regime`, `trend_regime` | Market environment context |

### Action Space (7 actions)

| Action | Description | Size Multiplier |
|--------|-------------|-----------------|
| `HOLD` | No action | 0% |
| `BUY_SMALL` | Long UP token | 25% |
| `BUY_MEDIUM` | Long UP token | 50% |
| `BUY_LARGE` | Long UP token | 100% |
| `SELL_SMALL` | Long DOWN token | 25% |
| `SELL_MEDIUM` | Long DOWN token | 50% |
| `SELL_LARGE` | Long DOWN token | 100% |

### Reward Shaping

Dense reward signal for sample efficiency:

1. **Unrealized PnL delta** - Main learning signal
2. **Transaction cost penalty** - Discourages overtrading
3. **Spread cost on entry** - Accounts for slippage
4. **Expiry urgency penalty** - Encourages closing before expiry
5. **Momentum alignment bonus** - Rewards trading with trend
6. **Size-confidence matching** - Rewards appropriate position sizing

## Data Sources

### Real-time Streams
- **Binance WebSocket** - Spot prices for BTC, ETH, SOL, XRP
- **Binance Futures WebSocket** - Trade flow, CVD, large trade detection
- **Polymarket CLOB WebSocket** - Orderbook depth, bid/ask spreads

### Polling Data
- **Binance Futures API** - Funding rates, open interest, mark price
- **Binance Klines** - Multi-timeframe returns (1m, 5m, 10m, 15m, 1h)
- **Polymarket API** - Active markets, token prices

## Usage

### Training Mode
```bash
# Train RL agent with $100 position sizing
python run.py --strategy rl --train --size 100

# Train with custom buffer size (more data per update)
python run.py --strategy rl --train --size 50
```

### Inference Mode
```bash
# Load trained model and run
python run.py --strategy rl --load rl_model --size 100
```

### Dashboard
```bash
# Run in separate terminal
python dashboard.py --port 5001
# Open http://localhost:5001
```

### Baseline Strategies
```bash
python run.py --strategy momentum   # Follow short-term trends
python run.py --strategy mean_revert # Fade extreme moves
python run.py --strategy fade_spike  # Fade large spikes
python run.py --strategy random      # Random baseline
```

## PPO Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_actor` | 3e-4 | Actor learning rate |
| `lr_critic` | 1e-3 | Critic learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `clip_epsilon` | 0.2 | PPO surrogate clipping range |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `buffer_size` | 2048 | Experience buffer before update |
| `batch_size` | 128 | Mini-batch size for SGD |
| `n_epochs` | 10 | PPO epochs per update |
| `target_kl` | 0.02 | Early stopping KL threshold |
| `max_grad_norm` | 0.5 | Gradient clipping |

## Network Architecture

```
Actor (Policy Network)
├── Linear(18 → 128) + Tanh
├── Linear(128 → 128) + Tanh
└── Linear(128 → 7) + Softmax

Critic (Value Network)
├── Linear(18 → 128) + Tanh
├── Linear(128 → 128) + Tanh
└── Linear(128 → 1)
```

## Dashboard Features

- **Live market cards** - Probability, time remaining, velocity
- **Position tracking** - Current exposure with unrealized P&L
- **P&L chart** - Rolling performance visualization
- **RL metrics** - Buffer progress, policy/value loss, entropy, KL divergence
- **Trade feed** - Recent trade history with outcomes

## Requirements

```
mlx>=0.5.0        # Apple Silicon ML framework
websockets>=12.0  # Async WebSocket client
flask>=3.0.0      # Web framework
flask-socketio>=5.3.0  # Real-time dashboard
numpy>=1.24.0     # Numerical computing
requests>=2.31.0  # HTTP client
```

## Installation

```bash
cd experiments/03_polymarket
python -m venv venv
source venv/bin/activate
pip install mlx websockets flask flask-socketio numpy requests
```

## Paper Trading

This system is for **paper trading only**. It does not execute real trades on Polymarket. All positions are simulated using live market data.

To connect to real trading, you would need:
1. Polymarket CLOB API credentials
2. Wallet with USDC on Polygon
3. Order execution logic (not implemented)

## Performance Notes

- Markets refresh every 15 minutes
- Decision loop runs at ~2 Hz (500ms tick)
- PPO updates take ~1-2 seconds on M1/M2
- Dashboard updates every tick for responsiveness

## Known Limitations

- No real order execution
- Single-threaded decision loop
- No persistence of training state across restarts (save manually)
- Markets may have low liquidity during off-hours
