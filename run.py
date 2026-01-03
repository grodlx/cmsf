#!/usr/bin/env python3
"""
Unified runner for Polymarket trading strategies.
"""
import asyncio
import argparse
import copy
import sys
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, ".")
from helpers import get_15m_markets, BinanceStreamer, OrderbookStreamer, Market, FuturesStreamer, get_logger
from strategies import (
    Strategy, MarketState, Action,
    create_strategy, AVAILABLE_STRATEGIES,
    RLStrategy,
)

# Dashboard integration (optional)
try:
    from dashboard_cinematic import update_dashboard_state, update_rl_metrics, emit_rl_buffer, run_dashboard, emit_trade
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    def update_dashboard_state(**kwargs): pass
    def update_rl_metrics(metrics): pass
    def emit_rl_buffer(buffer_size, max_buffer=256, avg_reward=None): pass
    def emit_trade(action, asset, size=0, pnl=None): pass


@dataclass
class Position:
    """Track position for a market."""
    asset: str
    side: Optional[str] = None
    size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_prob: float = 0.0
    time_remaining_at_entry: float = 0.0


class TradingEngine:
    """
    Paper trading engine with strategy harness.
    """

    def __init__(self, strategy: Strategy, trade_size: float = 10.0):
        self.strategy = strategy
        self.trade_size = trade_size

        # Streamers
        self.price_streamer = BinanceStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.orderbook_streamer = OrderbookStreamer()
        self.futures_streamer = FuturesStreamer(["BTC", "ETH", "SOL", "XRP"])

        # State
        self.markets: Dict[str, Market] = {}
        self.positions: Dict[str, Position] = {}
        self.states: Dict[str, MarketState] = {}
        self.prev_states: Dict[str, MarketState] = {}  # For RL transitions
        self.open_prices: Dict[str, float] = {}  # Binance price at market open
        self.running = False

        # Stats
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        # Pending rewards for RL (set on position close)
        self.pending_rewards: Dict[str, float] = {}

        # Logger (for RL training)
        self.logger = get_logger() if isinstance(strategy, RLStrategy) else None

    def refresh_markets(self):
        """Find active 15-min markets."""
        print("\n" + "=" * 60)
        print(f"STRATEGY: {self.strategy.name.upper()}")
        print("=" * 60)

        markets = get_15m_markets(assets=["BTC", "ETH", "SOL", "XRP"])
        now = datetime.now(timezone.utc)

        self.markets.clear()
        self.states.clear()

        for m in markets:
            mins_left = (m.end_time - now).total_seconds() / 60
            if mins_left < 0.5:
                continue

            print(f"\n{m.asset} 15m | {mins_left:.1f}m left")
            print(f"  UP: {m.price_up:.3f} | DOWN: {m.price_down:.3f}")

            self.markets[m.condition_id] = m
            self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)

            self.states[m.condition_id] = MarketState(
                asset=m.asset,
                prob=m.price_up,
                time_remaining=mins_left / 15.0,
            )

            if m.condition_id not in self.positions:
                self.positions[m.condition_id] = Position(asset=m.asset)

            current_price = self.price_streamer.get_price(m.asset)
            if current_price > 0:
                self.open_prices[m.condition_id] = current_price

        if not self.markets:
            print("\nNo active markets!")
        else:
            active_cids = set(self.markets.keys())
            self.orderbook_streamer.clear_stale(active_cids)

    def execute_action(self, cid: str, action: Action, state: MarketState):
        """Execute paper trade with 1% slippage and unified reporting."""
        if action == Action.HOLD:
            return

        pos = self.positions.get(cid)
        if not pos:
            return

        price = state.prob
        trade_amount = self.trade_size * action.size_multiplier
        
        FEE = 1.01      # Nakup o 1% draz
        EXIT_FEE = 0.99 # Prodej o 1% levneji

        if pos.size > 0:
            hold_duration = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()
            if hold_duration < 30:
                return

            if (action.is_sell and pos.side == "UP") or (action.is_buy and pos.side == "DOWN"):
                shares = pos.size / pos.entry_price
                
                if pos.side == "UP":
                    effective_exit_price = price * EXIT_FEE
                else:
                    effective_exit_price = (1 - price) * EXIT_FEE
                
                pnl = (effective_exit_price - pos.entry_price) * shares
                self._record_trade(pos, effective_exit_price, pnl, f"CLOSE {pos.side}", cid=cid)
                
                self.pending_rewards[cid] = pnl
                pos.size = 0
                pos.side = None
                return

        if pos.size == 0:
            size_label = {0.25: "SM", 0.5: "MD", 1.0: "LG"}.get(action.size_multiplier, "MD")

            if action.is_buy:
                pos.side = "UP"
                pos.entry_price = price * FEE
            elif action.is_sell:
                pos.side = "DOWN"
                pos.entry_price = (1 - price) * FEE

            pos.size = trade_amount
            pos.entry_time = datetime.now(timezone.utc)
            pos.entry_prob = price
            pos.time_remaining_at_entry = state.time_remaining
            
            print(f"    OPEN {pos.asset} {pos.side} @ {pos.entry_price:.3f} (Slippage included)")
            emit_trade(f"BUY_{size_label}" if action.is_buy else f"SELL_{size_label}", pos.asset, pos.size)

    def _record_trade(self, pos: Position, price: float, pnl: float, action: str, cid: str = None):
        """Unified distribution of trade results."""
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1

        print(f"    {action} {pos.asset} @ {price:.3f} | Realized PnL: ${pnl:+.2f}")
        emit_trade(action, pos.asset, pos.size, pnl)
        self._update_dashboard_only()

        if self.logger and pos.entry_time:
            duration = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()
            self.logger.log_trade(
                asset=pos.asset,
                action=action,
                side=pos.side or "UNKNOWN",
                entry_price=pos.entry_price,
                exit_price=price,
                size=pos.size,
                pnl=pnl,
                duration_sec=duration,
                time_remaining=pos.time_remaining_at_entry,
                prob_at_entry=pos.entry_prob,
                prob_at_exit=price,
                condition_id=cid
            )

    def _compute_step_reward(self, cid: str, state: MarketState, action: Action, pos: Position) -> float:
        return self.pending_rewards.pop(cid, 0.0)

    def close_all_positions(self):
        """Uzavre vsechny pozice s 1% slippage pri ukonceni programu."""
        EXIT_FEE = 0.99
        for cid, pos in self.positions.items():
            if pos.size > 0:
                state = self.states.get(cid)
                if state:
                    price = state.prob
                    shares = pos.size / pos.entry_price
                    if pos.side == "UP":
                        eff_exit = price * EXIT_FEE
                    else:
                        eff_exit = (1 - price) * EXIT_FEE
                    pnl = (eff_exit - pos.entry_price) * shares
                    self._record_trade(pos, eff_exit, pnl, f"FORCE CLOSE {pos.side}", cid=cid)
                    pos.size = 0
                    pos.side = None

    async def decision_loop(self):
        """Main trading loop."""
        tick = 0
        tick_interval = 0.5
        while self.running:
            await asyncio.sleep(tick_interval)
            tick += 1
            now = datetime.now(timezone.utc)

            expired = [cid for cid, m in self.markets.items() if m.end_time <= now]
            for cid in expired:
                print(f"\n  EXPIRED: {self.markets[cid].asset}")
                if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                    state = self.states.get(cid)
                    prev_state = self.prev_states.get(cid)
                    pos = self.positions.get(cid)
                    if state and prev_state:
                        terminal_reward = state.position_pnl if pos and pos.size > 0 else 0.0
                        self.strategy.store(prev_state, Action.HOLD, terminal_reward, state, done=True)
                    if cid in self.prev_states:
                        del self.prev_states[cid]
                del self.markets[cid]

            if not self.markets:
                print("\nAll markets expired. Refreshing...")
                self.close_all_positions()
                self.refresh_markets()
                if not self.markets:
                    await asyncio.sleep(30)
                continue

            for cid, m in self.markets.items():
                state = self.states.get(cid)
                if not state: continue

                ob = self.orderbook_streamer.get_orderbook(cid, "UP")
                if ob and ob.mid_price:
                    state.prob = ob.mid_price
                    state.prob_history.append(ob.mid_price)
                    if len(state.prob_history) > 100:
                        state.prob_history = state.prob_history[-100:]
                    state.best_bid = ob.best_bid or 0.0
                    state.best_ask = ob.best_ask or 0.0
                    state.spread = ob.spread or 0.0

                    if ob.bids and ob.asks:
                        bid_vol_l1 = ob.bids[0][1] if ob.bids else 0
                        ask_vol_l1 = ob.asks[0][1] if ob.asks else 0
                        total_l1 = bid_vol_l1 + ask_vol_l1
                        state.order_book_imbalance_l1 = (bid_vol_l1 - ask_vol_l1) / total_l1 if total_l1 > 0 else 0.0
                        bid_vol_l5 = sum(size for _, size in ob.bids[:5])
                        ask_vol_l5 = sum(size for _, size in ob.asks[:5])
                        total_l5 = bid_vol_l5 + ask_vol_l5
                        state.order_book_imbalance_l5 = (bid_vol_l5 - ask_vol_l5) / total_l5 if total_l5 > 0 else 0.0

                binance_price = self.price_streamer.get_price(m.asset)
                state.binance_price = binance_price
                open_price = self.open_prices.get(cid, binance_price)
                if open_price > 0:
                    state.binance_change = (binance_price - open_price) / open_price

                futures = self.futures_streamer.get_state(m.asset)
                if futures:
                    old_cvd = state.cvd
                    state.cvd = futures.cvd
                    state.cvd_acceleration = (futures.cvd - old_cvd) / 1e6 if old_cvd != 0 else 0.0
                    state.trade_flow_imbalance = futures.trade_flow_imbalance
                    state.returns_1m = futures.returns_1m
                    state.returns_5m = futures.returns_5m
                    state.returns_10m = futures.returns_10m
                    state.trade_intensity = futures.trade_intensity
                    state.large_trade_flag = futures.large_trade_flag
                    state.realized_vol_5m = futures.realized_vol_1h / 3.5 if futures.realized_vol_1h > 0 else 0.0
                    state.vol_expansion = futures.vol_ratio - 1.0
                    state.vol_regime = 1.0 if futures.realized_vol_1h > 0.01 else 0.0
                    state.trend_regime = 1.0 if abs(futures.returns_1h) > 0.005 else 0.0

                state.time_remaining = (m.end_time - now).total_seconds() / 900
                pos = self.positions.get(cid)
                if pos and pos.size > 0:
                    state.has_position = True
                    state.position_side = pos.side
                    shares = pos.size / pos.entry_price
                    current_val = state.prob if pos.side == "UP" else (1 - state.prob)
                    state.position_pnl = (current_val - pos.entry_price) * shares
                else:
                    state.has_position = False
                    state.position_pnl = 0.0

                action = self.strategy.act(state)

                if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                    prev_state = self.prev_states.get(cid)
                    if prev_state:
                        step_reward = self._compute_step_reward(cid, state, action, pos)
                        self.strategy.store(prev_state, action, step_reward, state, done=False)
                    self.prev_states[cid] = copy.deepcopy(state)

                if action != Action.HOLD:
                    self.execute_action(cid, action, state)

            if tick % 10 == 0:
                self.print_status()
            else:
                self._update_dashboard_only()

            if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                buffer_size = len(self.strategy.experiences)
                avg_reward = None
                if buffer_size > 0:
                    recent_rewards = [exp.reward for exp in self.strategy.experiences[-50:]]
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                emit_rl_buffer(buffer_size, self.strategy.buffer_size, avg_reward)

                if buffer_size >= self.strategy.buffer_size:
                    buffer_rewards = [exp.reward for exp in self.strategy.experiences]
                    metrics = self.strategy.update()
                    if metrics:
                        print(f"  [RL] update metrics: {metrics}")
                        update_rl_metrics(metrics)
                        if self.logger:
                            self.logger.log_update(metrics=metrics, buffer_rewards=buffer_rewards, cumulative_pnl=self.total_pnl, cumulative_trades=self.trade_count, cumulative_wins=self.win_count)

    def _update_dashboard_only(self):
        """
        Synchronizuje aktualni stav obchodovani do weboveho dashboardu.
        """
        now = datetime.now(timezone.utc)
        dashboard_markets = {}
        dashboard_positions = {}

        for cid, m in self.markets.items():
            state = self.states.get(cid)
            pos = self.positions.get(cid)
            
            if state:
                mins_left = (m.end_time - now).total_seconds() / 60
                vel = state._velocity() if hasattr(state, '_velocity') else 0.0
                
                dashboard_markets[cid] = {
                    'asset': m.asset,
                    'prob': state.prob,
                    'time_left': mins_left,
                    'velocity': vel,
                }

                if pos and pos.size > 0:
                    current_val = state.prob if pos.side == "UP" else (1 - state.prob)
                    shares = pos.size / pos.entry_price
                    unrealized_pnl = (current_val - pos.entry_price) * shares

                    dashboard_positions[cid] = {
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'unrealized_pnl': unrealized_pnl
                    }

        if DASHBOARD_AVAILABLE:
            update_dashboard_state(
                strategy_name=self.strategy.name,
                total_pnl=self.total_pnl,
                trade_count=self.trade_count,
                win_count=self.win_count,
                positions=dashboard_positions,
                markets=dashboard_markets
            )

    def print_status(self):
        """Print current status."""
        now = datetime.now(timezone.utc)
        win_rate = self.win_count / max(1, self.trade_count) * 100
        print(f"\n[{now.strftime('%H:%M:%S')}] {self.strategy.name.upper()}")
        print(f"  PnL: ${self.total_pnl:+.2f} | Trades: {self.trade_count} | Win: {win_rate:.0f}%")
        self._update_dashboard_only()

    def print_final_stats(self):
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Strategy: {self.strategy.name}")
        print(f"Total PnL: ${self.total_pnl:+.2f}")
        print(f"Trades: {self.trade_count}")
        print(f"Win Rate: {self.win_count / max(1, self.trade_count) * 100:.1f}%")

    async def run(self):
        self.running = True
        self.refresh_markets()
        if not self.markets:
            print("No markets to trade!")
            return
        tasks = [
            self.price_streamer.stream(),
            self.orderbook_streamer.stream(),
            self.futures_streamer.stream(),
            self.decision_loop(),
        ]
        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            self.running = False
            self.price_streamer.stop()
            self.orderbook_streamer.stop()
            self.futures_streamer.stop()
            self.close_all_positions()
            self.print_final_stats()
            if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                self.strategy.save("rl_model")

async def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading")
    parser.add_argument("strategy", nargs="?", choices=AVAILABLE_STRATEGIES, help="Strategy to run")
    parser.add_argument("--train", action="store_true", help="Enable training for RL")
    parser.add_argument("--size", type=float, default=10.0, help="Trade size")
    parser.add_argument("--load", type=str, help="Load RL model")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Port")

    args = parser.parse_args()
    if not args.strategy:
        print("Available strategies:", AVAILABLE_STRATEGIES)
        return

    if args.dashboard and DASHBOARD_AVAILABLE:
        threading.Thread(target=run_dashboard, kwargs={'port': args.port}, daemon=True).start()

    strategy = create_strategy(args.strategy)
    if isinstance(strategy, RLStrategy):
        if args.load: strategy.load(args.load)
        if args.train: strategy.train()
        else: strategy.eval()

    engine = TradingEngine(strategy, trade_size=args.size)
    await engine.run()

if __name__ == "__main__":
    asyncio.run(main())
