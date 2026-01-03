#!/usr/bin/env python3
import asyncio
import argparse
import copy
import sys
import threading
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import deque
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
    tp_level: float = 0.0
    sl_level: float = 0.0

class TradingEngine:
    def __init__(self, strategy: Strategy, trade_size: float = 10.0):
        self.strategy = strategy
        self.trade_size = trade_size
        self.price_streamer = BinanceStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.orderbook_streamer = OrderbookStreamer()
        self.futures_streamer = FuturesStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.markets, self.positions, self.states = {}, {}, {}
        
        # --- FILTRY A BUFFERY ---
        self.prob_buffers = {}    
        self.atr_buffers = {}     
        self.cooldowns = {}       
        self.prev_states = {}
        self.pending_rewards = {}
        
        self.total_pnl, self.trade_count, self.win_count = 0.0, 0, 0
        self.running = False
        
        os.makedirs("logs", exist_ok=True)
        self.tick_log_path = f"logs/market_ticks_hft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._init_tick_log()

    def _init_tick_log(self):
        with open(self.tick_log_path, "w") as f:
            f.write("timestamp,asset,prob,binance_price,vol_5m,spread\n")

    def execute_action(self, cid: str, action: Action, state: MarketState):
        pos = self.positions.get(cid)
        if not pos: return
        now_ts = time.time()

        # 1. FILTRACE SIGNÁLU
        if cid not in self.prob_buffers:
            self.prob_buffers[cid] = deque(maxlen=5) 
            self.atr_buffers[cid] = deque(maxlen=20)

        self.prob_buffers[cid].append(state.prob)
        self.atr_buffers[cid].append(state.realized_vol_5m if state.realized_vol_5m > 0 else 0.005)
        
        smoothed_prob = sum(self.prob_buffers[cid]) / len(self.prob_buffers[cid])
        avg_atr = sum(self.atr_buffers[cid]) / len(self.atr_buffers[cid])

        FEE, EXIT_FEE = 1.01, 0.99 

        # 2. LOGIKA VÝSTUPU
        if pos.size > 0:
            curr_val = state.prob if pos.side == "UP" else (1 - state.prob)
            is_tp = curr_val >= pos.tp_level
            is_sl = curr_val <= pos.sl_level
            
            if is_tp or is_sl:
                shares = pos.size / pos.entry_price
                eff_exit = curr_val * EXIT_FEE
                pnl = (eff_exit - pos.entry_price) * shares
                
                reason = "TP" if is_tp else "SL"
                self._record_trade(pos, eff_exit, pnl, f"CLOSE {pos.side} ({reason})", cid=cid)
                self.pending_rewards[cid] = pnl
                pos.size, pos.side = 0, None
                self.cooldowns[cid] = now_ts + 15 
            return

        # 3. LOGIKA VSTUPU (REVERSAL)
        if pos.size == 0 and now_ts > self.cooldowns.get(cid, 0):
            if len(self.prob_buffers[cid]) < 5: return

            UPPER_THR = 0.62
            LOWER_THR = 0.38

            if smoothed_prob > UPPER_THR:
                pos.side = "DOWN"
                pos.entry_price = (1 - state.prob) * FEE
                pos.tp_level = pos.entry_price + (avg_atr * 1.5)
                pos.sl_level = pos.entry_price - (avg_atr * 2.5)
                pos.size = self.trade_size * action.size_multiplier
                pos.entry_time = datetime.now(timezone.utc)
                print(f"📉 [REV] SHORT {pos.asset} @ {pos.entry_price:.3f}")

            elif smoothed_prob < LOWER_THR:
                pos.side = "UP"
                pos.entry_price = state.prob * FEE
                pos.tp_level = pos.entry_price + (avg_atr * 1.5)
                pos.sl_level = pos.entry_price - (avg_atr * 2.5)
                pos.size = self.trade_size * action.size_multiplier
                pos.entry_time = datetime.now(timezone.utc)
                print(f"📈 [REV] LONG {pos.asset} @ {pos.entry_price:.3f}")

    def _compute_step_reward(self, cid: str) -> float:
        pnl = self.pending_rewards.pop(cid, 0.0)
        if pnl == 0: return 0.0
        penalty = -0.15 
        return (pnl + penalty) if pnl > 0 else (pnl * 1.5 + penalty)

    def _record_trade(self, pos: Position, price: float, pnl: float, action: str, cid: str = None):
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0: self.win_count += 1
        print(f"    {action} {pos.asset} @ {price:.3f} | PnL: ${pnl:+.2f}")
        emit_trade(action, pos.asset, pos.size, pnl)

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

    def _update_dashboard_only(self):
        if not DASHBOARD_AVAILABLE: return
        try:
            now = datetime.now(timezone.utc)
            d_m, d_p = {}, {}
            for cid, m in self.markets.items():
                s = self.states.get(cid)
                p = self.positions.get(cid)
                if s:
                    d_m[cid] = {'asset': m.asset, 'prob': s.prob, 'time_left': (m.end_time-now).total_seconds()/60}
                    if p and p.size > 0:
                        cur = s.prob if p.side == "UP" else (1 - s.prob)
                        d_p[cid] = {'side': p.side, 'size': p.size, 'entry_price': p.entry_price, 'unrealized_pnl': (cur - p.entry_price) * (p.size/p.entry_price)}
            update_dashboard_state(strategy_name=self.strategy.name, total_pnl=self.total_pnl, trade_count=self.trade_count, win_count=self.win_count, positions=d_p, markets=d_m)
        except: pass

    async def decision_loop(self):
        tick = 0
        while self.running:
            try:
                await asyncio.sleep(0.5)
                tick += 1
                now = datetime.now(timezone.utc)
                if not self.markets: self.refresh_markets(); await asyncio.sleep(5); continue
                
                for cid, m in self.markets.items():
                    if m.end_time <= now: continue
                    state = self.states.get(cid)
                    ob = self.orderbook_streamer.get_orderbook(cid, "UP")
                    if ob and ob.mid_price: state.prob = ob.mid_price
                    
                    action = self.strategy.act(state)
                    
                    if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                        prev = self.prev_states.get(cid)
                        if prev: 
                            reward = self._compute_step_reward(cid)
                            self.strategy.store(prev, action, reward, state, False)
                        self.prev_states[cid] = copy.deepcopy(state)
                    
                    self.execute_action(cid, action, state)
                    
                if tick % 40 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] PnL: ${self.total_pnl:+.2f} | Trades: {self.trade_count}")
                    self._update_dashboard_only()
            except Exception as e: print(f"Loop Error: {e}")

    def close_all_positions(self):
        for cid, pos in self.positions.items():
            if pos.size > 0: self._record_trade(pos, 0, 0, "FORCE CLOSE"); pos.size = 0

    async def run(self):
        self.running = True
        self.refresh_markets()
        tasks = [asyncio.create_task(self.price_streamer.stream()), asyncio.create_task(self.orderbook_streamer.stream()),
                 asyncio.create_task(self.futures_streamer.stream()), asyncio.create_task(self.decision_loop())]
        try: await asyncio.gather(*tasks)
        except: pass
        finally:
            self.running = False
            for t in tasks: t.cancel()
            self.close_all_positions()
            if isinstance(self.strategy, RLStrategy) and self.strategy.training: self.strategy.save("rl_model")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=AVAILABLE_STRATEGIES)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--size", type=float, default=10.0)
    parser.add_argument("--dashboard", action="store_true")
    args = parser.parse_args()

    if args.dashboard and DASHBOARD_AVAILABLE:
        threading.Thread(target=run_dashboard, kwargs={'port': 5050}, daemon=True).start()

    strategy = create_strategy(args.strategy)
    if isinstance(strategy, RLStrategy) and args.train:
        print("🤖 RL TRAINING MODE ACTIVE")
        strategy.train()
    
    engine = TradingEngine(strategy, trade_size=args.size)
    await engine.run()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
