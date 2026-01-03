#!/usr/bin/env python3
import asyncio
import argparse
import copy
import sys
import threading
import os
import signal
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

# Dashboard integrace
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
    asset: str
    side: Optional[str] = None
    size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_prob: float = 0.0
    time_remaining_at_entry: float = 0.0

class TradingEngine:
    def __init__(self, strategy: Strategy, trade_size: float = 10.0):
        self.strategy = strategy
        self.trade_size = trade_size
        self.price_streamer = BinanceStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.orderbook_streamer = OrderbookStreamer()
        self.futures_streamer = FuturesStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.markets, self.positions, self.states = {}, {}, {}
        self.prev_states, self.open_prices = {}, {}
        self.total_pnl, self.trade_count, self.win_count = 0.0, 0, 0
        self.pending_rewards = {}
        self.running = False
        self.logger = get_logger() if isinstance(strategy, RLStrategy) else None
        
        # Inicializace Tick Logu (HFT režim)
        os.makedirs("logs", exist_ok=True)
        self.tick_log_path = f"logs/market_ticks_hft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._init_tick_log()

    def _init_tick_log(self):
        with open(self.tick_log_path, "w") as f:
            f.write("timestamp,asset,prob,binance_price,vol_5m,spread,imbalance_l1,trade_intensity\n")

    def _log_market_tick(self, cid, state):
        try:
            now = datetime.now(timezone.utc).isoformat()
            with open(self.tick_log_path, "a") as f:
                f.write(f"{now},{state.asset},{state.prob:.5f},{state.binance_price:.2f},"
                        f"{state.realized_vol_5m:.6f},{state.spread:.6f},"
                        f"{state.order_book_imbalance_l1:.4f},{state.trade_intensity:.2f}\n")
        except: pass

    def refresh_markets(self):
        markets = get_15m_markets(assets=["BTC", "ETH", "SOL", "XRP"])
        now = datetime.now(timezone.utc)
        self.markets.clear()
        self.states.clear()
        for m in markets:
            mins_left = (m.end_time - now).total_seconds() / 60
            if mins_left < 0.5: continue
            self.markets[m.condition_id] = m
            self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)
            self.states[m.condition_id] = MarketState(asset=m.asset, prob=m.price_up, time_remaining=mins_left / 15.0)
            if m.condition_id not in self.positions:
                self.positions[m.condition_id] = Position(asset=m.asset)
            price = self.price_streamer.get_price(m.asset)
            if price > 0: self.open_prices[m.condition_id] = price

    def execute_action(self, cid: str, action: Action, state: MarketState):
        if action == Action.HOLD: return
        pos = self.positions.get(cid)
        if not pos: return

        prob_dist = abs(state.prob - 0.50)
        if pos.size == 0 and prob_dist < 0.04: return

        FEE, EXIT_FEE = 1.01, 0.99
        if pos.size > 0:
            if (action.is_sell and pos.side == "UP") or (action.is_buy and pos.side == "DOWN"):
                shares = pos.size / pos.entry_price
                eff_exit = (state.prob if pos.side == "UP" else (1 - state.prob)) * EXIT_FEE
                pnl = (eff_exit - pos.entry_price) * shares
                self._record_trade(pos, eff_exit, pnl, f"CLOSE {pos.side}", cid=cid)
                self.pending_rewards[cid] = pnl
                pos.size, pos.side = 0, None
                return

        if pos.size == 0:
            if action.is_buy:
                pos.side, pos.entry_price = "UP", state.prob * FEE
            else:
                pos.side, pos.entry_price = "DOWN", (1 - state.prob) * FEE
            pos.size, pos.entry_time = self.trade_size * action.size_multiplier, datetime.now(timezone.utc)
            print(f"    OPEN {pos.asset} {pos.side} @ {pos.entry_price:.3f}")

    def _record_trade(self, pos: Position, price: float, pnl: float, action: str, cid: str = None):
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0: self.win_count += 1
        print(f"    {action} {pos.asset} @ {price:.3f} | Realized PnL: ${pnl:+.2f}")
        emit_trade(action, pos.asset, pos.size, pnl)
        self._update_dashboard_only()

    def _compute_step_reward(self, cid: str, state: MarketState, action: Action, pos: Position) -> float:
        pnl = self.pending_rewards.pop(cid, 0.0)
        if pnl == 0: return 0.0
        penalty = -0.15 
        return (pnl + penalty) if pnl > 0 else (pnl * 1.5 + penalty)

    def close_all_positions(self):
        EXIT_FEE = 0.99
        for cid, pos in self.positions.items():
            if pos.size > 0 and cid in self.states:
                state = self.states[cid]
                eff_exit = (state.prob if pos.side == "UP" else (1 - state.prob)) * EXIT_FEE
                pnl = (eff_exit - pos.entry_price) * (pos.size / pos.entry_price)
                self._record_trade(pos, eff_exit, pnl, "FORCE CLOSE", cid=cid)
                pos.size = 0

    def _update_dashboard_only(self):
        try:
            now = datetime.now(timezone.utc)
            dashboard_m, dashboard_p = {}, {}
            for cid, m in self.markets.items():
                state = self.states.get(cid)
                pos = self.positions.get(cid)
                if state:
                    dashboard_m[cid] = {'asset': m.asset, 'prob': state.prob, 'time_left': (m.end_time - now).total_seconds()/60, 'velocity': 0}
                    if pos and pos.size > 0:
                        cur = state.prob if pos.side == "UP" else (1 - state.prob)
                        dashboard_p[cid] = {'side': pos.side, 'size': pos.size, 'entry_price': pos.entry_price, 'unrealized_pnl': (cur - pos.entry_price) * (pos.size/pos.entry_price)}
            if DASHBOARD_AVAILABLE:
                update_dashboard_state(strategy_name=self.strategy.name, total_pnl=self.total_pnl, trade_count=self.trade_count, win_count=self.win_count, positions=dashboard_p, markets=dashboard_m)
        except: pass

    async def decision_loop(self):
        tick, tick_interval = 0, 0.5 # HFT 0.5s
        while self.running:
            try:
                await asyncio.sleep(tick_interval)
                tick += 1
                now = datetime.now(timezone.utc)
                
                for cid in [c for c, m in self.markets.items() if m.end_time <= now]:
                    del self.markets[cid]
                
                if not self.markets:
                    self.refresh_markets()
                    await asyncio.sleep(10); continue
                    
                for cid, m in self.markets.items():
                    state = self.states.get(cid)
                    ob = self.orderbook_streamer.get_orderbook(cid, "UP")
                    if ob and ob.mid_price: 
                        state.prob = ob.mid_price
                        state.spread = ob.spread or 0.0
                    
                    self._log_market_tick(cid, state)
                    
                    action = self.strategy.act(state)
                    if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                        prev = self.prev_states.get(cid)
                        if prev: self.strategy.store(prev, action, self._compute_step_reward(cid, state, action, self.positions[cid]), state, done=False)
                        self.prev_states[cid] = copy.deepcopy(state)
                    
                    self.execute_action(cid, action, state)
                    
                if tick % 20 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] PnL: ${self.total_pnl:+.2f} | Trades: {self.trade_count}")
            except asyncio.CancelledError: break
            except Exception as e:
                print(f"Error in loop: {e}")

    async def run(self):
        self.running = True
        self.refresh_markets()
        tasks = [
            asyncio.create_task(self.price_streamer.stream()),
            asyncio.create_task(self.orderbook_streamer.stream()),
            asyncio.create_task(self.futures_streamer.stream()),
            asyncio.create_task(self.decision_loop())
        ]
        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nShutting down gracefully...")
        finally:
            self.running = False
            # Zrušení úloh před vypnutím loopu
            for t in tasks: t.cancel()
            self.close_all_positions()
            if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                print("Saving RL model...")
                self.strategy.save("rl_model")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=AVAILABLE_STRATEGIES)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--size", type=float, default=10.0)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    if args.dashboard and DASHBOARD_AVAILABLE:
        threading.Thread(target=run_dashboard, kwargs={'port': args.port}, daemon=True).start()

    strategy = create_strategy(args.strategy)
    if isinstance(strategy, RLStrategy) and args.train: strategy.train()
    
    engine = TradingEngine(strategy, trade_size=args.size)
    await engine.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
