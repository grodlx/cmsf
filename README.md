# Cross-Market State Fusion

RL agents that exploit information lag between fast markets (Binance futures) and slow markets (Polymarket prediction markets) through real-time multi-source state fusion.

**[View the presentation (PDF)](cross-market-state-fusion.pdf)**

## What This Is

A PPO (Proximal Policy Optimization) agent that paper trades Polymarket's 15-minute binary crypto markets. The agent observes live data from Binance futures and Polymarket's orderbook, then learns to predict short-term price direction.

**Current status**: Paper trading only. The agent trains and makes decisions on live market data, but doesn't execute real orders.

## What This Proves

1. **RL can learn from sparse PnL signals** - The agent only gets reward when positions close. No intermediate feedback during the 15-minute window. Despite this sparsity, it learns profitable patterns (~$50K PnL, 2,500% ROI in Phase 5 with temporal architecture).

2. **Multi-source data fusion works** - Combining Binance futures order flow and Polymarket orderbook state into a single 18-dim observation gives the agent useful signal.

3. **Low win rate can be profitable** - The agent wins only 23% of trades but profits because binary markets have asymmetric payoffs. Buy at 0.40, win pays 0.60; lose costs 0.40.

4. **On-device training is viable** - MLX on Apple Silicon handles real-time PPO updates during live market hours without cloud GPU costs.

5. **Temporal context helps** - Processing the last 5 market states through a TemporalEncoder improves decision quality by capturing momentum and trend patterns.

**Important caveat**: Training uses share-based PnL, not actual binary outcomes. See Phase 4 below for why this matters.

## What This Doesn't Prove

1. **Live profitability** - Paper trading assumes instant fills at mid-price. Real trading faces latency, slippage, and market impact. Expect 20-50% performance degradation.

2. **Statistical significance** - A single session isn't enough to confirm edge. Could be variance. Needs weeks of out-of-sample testing.

3. **Scalability** - $500 positions already show some market impact. At larger sizes the agent's orders would move prices and consume liquidity faster than it can trade.

4. **Persistence of edge** - Markets adapt. If this strategy worked, others would copy it and arbitrage it away.

## Path to Live Trading

To move from paper to real:

1. **Execution layer** - Integrate Polymarket CLOB API for order placement
2. **Slippage modeling** - Simulate walking the book at realistic sizes
3. **Latency compensation** - Account for 50-200ms round-trip to Polymarket
4. **Risk management** - Position limits, drawdown stops, exposure caps
5. **Extended validation** - Weeks of paper trading across market regimes

See [TRAINING_JOURNAL.md](TRAINING_JOURNAL.md) for detailed training analysis.

---

## Training Status

| Phase | Size | Updates | Trades | PnL | Win Rate | Entropy | ROI |
|-------|------|---------|--------|-----|----------|---------|-----|
| 1 (Shaped rewards) | $5 | 36 | 1,545 | $3.90 | 20.2% | 0.36 (collapsed) | - |
| 2 (Prob-based) | $5 | 36 | 3,330 | $10.93 ($11*) | 21.2% | 1.05 (healthy) | 55% (57%*) |
| 3 (Scaled up) | $50 | 36 | 4,133 | $23.10 ($76*) | 15.6% | 0.97 (healthy) | 12% (38%*) |
| 4 (Share-based) | $500 | 46 | 4,873 | $3,392 | 19.0% | 1.08 (healthy) | 170% |
| 5 (LACUNA) | $500 | - | 34,730 | ~$50K | 23.3% | 1.05 (healthy) | 2,500% |

**Capital**: Position size × 4 markets = max exposure ($20 Phase 2, $200 Phase 3, $2000 Phase 4-5)

*Phases 2-3 used probability-based PnL: `(exit - entry) × dollars`. Phase 4 uses share-based PnL: `(exit - entry) × shares`. The starred values show Phases 2-3 recalculated with share-based formula for comparison.

See [TRAINING_JOURNAL.md](TRAINING_JOURNAL.md) for detailed phase-by-phase analysis with charts.

---

## Architecture

```
├── run.py                    # Main trading engine
├── dashboard.py              # Real-time web dashboard
├── strategies/
│   ├── base.py               # Action, MarketState, Strategy base classes
│   ├── rl_mlx.py             # PPO implementation (MLX)
│   ├── momentum.py           # Momentum baseline
│   ├── mean_revert.py        # Mean reversion baseline
│   └── fade_spike.py         # Spike fading baseline
└── helpers/
    ├── polymarket_api.py     # Polymarket REST API
    ├── binance_wss.py        # Binance WebSocket (spot price reference)
    ├── binance_futures.py    # Futures data (returns, order flow, CVD)
    └── orderbook_wss.py      # Polymarket CLOB WebSocket
```

## State Space (18 dimensions)

| Category | Features | Source |
|----------|----------|--------|
| Momentum | `returns_1m`, `returns_5m`, `returns_10m` | Binance futures |
| Order Flow | `ob_imbalance_l1`, `ob_imbalance_l5`, `trade_flow`, `cvd_accel` | Binance futures |
| Microstructure | `spread_pct`, `trade_intensity`, `large_trade_flag` | Polymarket CLOB |
| Volatility | `vol_5m`, `vol_expansion` | Polymarket (local), Binance futures |
| Position | `has_position`, `position_side`, `position_pnl`, `time_remaining` | Internal |
| Regime | `vol_regime`, `trend_regime` | Derived |

**Note**: Returns and spread are scaled by 100x. CVD acceleration is divided by 1e6. `vol_5m` is computed locally from Polymarket prob history; `vol_expansion` comes from Binance futures.

## Action Space

| Action | Description |
|--------|-------------|
| HOLD (0) | No action |
| BUY (1) | Long UP token (bet price goes up) |
| SELL (2) | Long DOWN token (bet price goes down) |

Fixed 50% position sizing. Originally had 7 actions with variable sizing (25/50/100%), simplified to reduce complexity.

## Network (Phase 5: Temporal Architecture)

```
TemporalEncoder: (history_len × 18) → 64 → LayerNorm → tanh → 32

Actor:  [current_state(18) + temporal_features(32)] = 50 → 64 → LN → tanh → 64 → LN → tanh → 3 (softmax)
Critic: [current_state(18) + temporal_features(32)] = 50 → 96 → LN → tanh → 96 → LN → tanh → 1
```

**Key changes from Phase 4**:
- **Temporal processing**: TemporalEncoder compresses last 5 states into momentum/trend features
- **Asymmetric architecture**: Larger critic (96 units) vs actor (64 units) for better value estimation
- **Feature normalization**: All 18 input features clamped to [-1, 1] range

## PPO Hyperparameters (Phase 5)

| Parameter | Value | Change from Phase 4 |
|-----------|-------|---------------------|
| `lr_actor` | 1e-4 | - |
| `lr_critic` | 3e-4 | - |
| `gamma` | 0.95 | ↓ from 0.99 (shorter horizon) |
| `gae_lambda` | 0.95 | - |
| `clip_epsilon` | 0.2 | - |
| `entropy_coef` | 0.03 | ↓ from 0.10 (allow sparse policy) |
| `buffer_size` | 256 | ↓ from 512 (faster adaptation) |
| `batch_size` | 64 | - |
| `n_epochs` | 10 | - |
| `history_len` | 5 | NEW |
| `temporal_dim` | 32 | NEW |

---

## Usage

```bash
# Training
python run.py --strategy rl --train --size 50

# Inference (load trained model)
python run.py --strategy rl --load rl_model --size 100

# Dashboard (separate terminal)
python dashboard.py --port 5001

# Baselines
python run.py --strategy momentum
python run.py --strategy mean_revert
python run.py --strategy random
```

## Requirements

```
mlx>=0.5.0
websockets>=12.0
flask>=3.0.0
flask-socketio>=5.3.0
numpy>=1.24.0
requests>=2.31.0
```

## Installation

```bash
cd experiments/03_polymarket
python -m venv venv
source venv/bin/activate
pip install mlx websockets flask flask-socketio numpy requests
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{cross_market_state_fusion,
  author = {Saravanan, Nikshep},
  organization = {HumanPlane},
  title = {Cross-Market State Fusion: Exploiting Information Lag with Multi-Source RL},
  year = {2025},
  url = {https://github.com/humanplane/cross-market-state-fusion}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
