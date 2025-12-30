# Cross-Market State Fusion

RL agents that exploit information lag between fast markets (Binance futures) and slow markets (Polymarket prediction markets) through real-time multi-source state fusion.

**[View the presentation (PDF)](cross-market-state-fusion.pdf)**

## What This Is

A PPO (Proximal Policy Optimization) agent that paper trades Polymarket's 15-minute binary crypto markets. The agent observes live data from Binance futures and Polymarket's orderbook, then learns to predict short-term price direction.

**Current status**: Paper trading only. The agent trains and makes decisions on live market data, but doesn't execute real orders.

## What This Proves

1. **RL can learn from sparse PnL signals** - The agent only gets reward when positions close. No intermediate feedback during the 15-minute window. Despite this sparsity, it learns profitable patterns (55% ROI in Phase 2 training).

2. **Multi-source data fusion works** - Combining Binance futures order flow and Polymarket orderbook state into a single 18-dim observation gives the agent useful signal.

3. **Low win rate can be profitable** - The agent wins only 21% of trades but profits because binary markets have asymmetric payoffs. Buy at 0.40, win pays 0.60; lose costs 0.40.

4. **On-device training is viable** - MLX on Apple Silicon handles real-time PPO updates during live market hours without cloud GPU costs.

**Important caveat**: Training uses probability-based PnL (exit_prob - entry_prob), not actual binary outcomes. This is a proxy signal - the agent learns "did probability move my way?" rather than "did I correctly predict UP vs DOWN?"

## What This Doesn't Prove

1. **Live profitability** - Paper trading assumes instant fills at mid-price. Real trading faces latency, slippage, and market impact. Expect 20-50% performance degradation.

2. **Statistical significance** - 72 updates over 2 hours isn't enough to confirm edge. Could be variance. Needs weeks of out-of-sample testing.

3. **Scalability** - $5 positions are invisible to the market. At $100+ the agent's own orders would move prices and consume liquidity.

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
| 2 (Pure PnL) | $5 | 36 | 3,330 | $10.93 | 21.2% | 1.05 (healthy) | 55% |
| 3 (Scaled up) | $50 | 36 | 4,133 | $23.10 | 15.6% | 0.97 (healthy) | 12% (44%*) |

**Capital**: Position size × 0.5 × 4 markets = max exposure ($20 Phase 2, $200 Phase 3)

**Key insight**: Phase 3 started with a -$64 drawdown in the first update (unlucky market timing). The agent recovered $87 over the next 35 updates to finish +$23. *44% ROI if measured from the drawdown trough - comparable to Phase 2's 55%.

### Phase 3 Analysis

![Phase 3 Trading Analysis](phase3_analysis.png)

**Key findings**:
- **Equity curve** (top-left): -$75 max drawdown early, then steady recovery. Policy didn't collapse under pressure.
- **PnL by asset** (top-right): XRP carried (+$44), ETH struggled (-$45). Same policy, different results per asset.
- **PnL by side** (top-right): UP bets (+$16) outperformed DOWN (-$7) despite similar win rates. Slight long bias works.
- **Entry distribution** (bottom-left): Agent favors extreme probabilities (near 0 or 1) - hunting asymmetric payoffs.
- **Duration vs PnL** (bottom-middle): Correlation 0.02 - trade length doesn't predict outcome. Quick flips ≈ longer holds.
- **Entry timing vs PnL** (bottom-right): Correlation 0.01 - early vs late entry in 15-min window doesn't matter. Agent reacts to market state, not time.

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

## Network

```
Actor:  18 → 128 (tanh) → 128 (tanh) → 3 (softmax)
Critic: 18 → 128 (tanh) → 128 (tanh) → 1
```

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| `lr_actor` | 1e-4 |
| `lr_critic` | 3e-4 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_epsilon` | 0.2 |
| `entropy_coef` | 0.10 |
| `buffer_size` | 512 |
| `batch_size` | 64 |
| `n_epochs` | 10 |

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
  author = {Humanplane},
  title = {Cross-Market State Fusion: RL Agents for Prediction Market Trading},
  year = {2025},
  url = {https://github.com/humanplane/cross-market-state-fusion}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
