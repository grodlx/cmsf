#!/usr/bin/env python3
import asyncio
import argparse
import copy
import sys
import threading
import os
import time
import logging
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

# Nastavení logování
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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
        
        self.markets: Dict[str, Market] = {}
        self.positions: Dict[str, Position] = {}
        self.states: Dict[str, MarketState] = {}
        
        self.prob_buffers = {}    
        self.atr_buffers = {}     
        self.cooldowns = {}       
        self.prev_states = {}
        self.pending_rewards = {}
        
        self.total_pnl, self.trade_count, self.win_count = 0.0, 0, 0
        self.running, self.last_market_refresh = False, 0

    def execute_action(self, cid: str, state: MarketState):
        pos = self.positions.get(cid)
        if not pos: return
        now_ts = time.time()

        # 1. FILTRACE SIGNÁLU (Smoothing)
        if cid not in self.prob_buffers:
            self.prob_buffers[cid] = deque(maxlen=5) 
            self.atr_buffers[cid] = deque(maxlen=20)

        self.prob_buffers[cid].append(state.prob)
        # Použití realized_vol_5m nebo defaultní hodnoty
        vol = getattr(state, 'realized_vol_5m', 0.005)
        if vol <= 0: vol = 0.005
        self.atr_buffers[cid].append(vol)
        
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
                
                reason = "TAKE_PROFIT" if is_tp else "STOP_LOSS"
                self._record_trade(pos, eff_exit, pnl, f"CLOSE {pos.side} ({reason})", cid=cid)
                
                self.pending_rewards[cid] = pnl
                pos.size, pos.side = 0, None
                self.cooldowns[cid] = now_ts + 10 # 10s cooldown
            return

        # 3. LOGIKA VSTUPU (REVERSAL MODE)
        if pos.size == 0 and now_ts > self.cooldowns.get(cid, 0):
            if len(self.prob_buffers[cid]) < 3: return

            UPPER_THR, LOWER_THR = 0.60, 0.40

            if smoothed_prob > UPPER_THR:
                pos.side = "DOWN"
                pos.entry_price = (1 - state.prob) * FEE
                pos.tp_level = pos.entry_price + (avg_atr * 1.3)
                pos.sl_level = pos.entry_price - (avg_atr * 2.2)
                pos.size = self.trade_size
                pos.entry_time = datetime.now(timezone.utc)
                logging.info(f"📉 [REV] SHORT {pos.asset} @ {pos.entry_price:.3f} (Prob: {smoothed_prob:.3f})")

            elif smoothed_prob < LOWER_THR:
                pos.side = "UP"
                pos.entry_price = state.prob * FEE
                pos.tp_level = pos.entry_price + (avg_atr * 1.3)
                pos.sl_level = pos.entry_price - (avg_atr * 2.2)
                pos.size = self.trade_size
                pos.entry_time = datetime.now(timezone.utc)
                logging.info(f"📈 [REV] LONG {pos.asset} @ {pos.entry_price:.3f} (Prob: {smoothed_prob:.3f})")

    def _record_trade(self, pos: Position, price: float, pnl: float, action: str, cid: str = None):
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0: self.win_count += 1
        logging.info(f"💰 {action} {pos.asset} PnL: ${pnl:+.2f} | Total: ${self.total_pnl:.2f}")
        emit_trade(action, pos.asset, pos.size, pnl)
        self._update_dashboard_only()

    def refresh_markets(self):
        try:
            markets = get_15m_markets(assets=["BTC", "ETH", "SOL", "XRP"])
            now = datetime.now(timezone.utc)
            self.markets.clear()
            for m in markets:
                mins_left = (m.end_time - now).total_seconds() / 60
                if mins_left < 0.5: continue
                
                self.markets[m.condition_id] = m
                self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)
                
                # OPRAVA: Inicializace MarketState s time_remaining
                if m.condition_id not in self.states:
                    self.states[m.condition_id] = MarketState(
                        asset=m.asset, 
                        prob=m.price_up, 
                        time_remaining=mins_left / 15.0
                    )
                else:
                    self.states[m.condition_id].time_remaining = mins_left / 15.0
                
                if m.condition_id not in self.positions:
                    self.positions[m.condition_id] = Position(asset=m.asset)
            logging.info(f"🔄 Markets refreshed: {len(self.markets)} active")
        except Exception as e:
            logging.error(f"❌ Refresh Error: {e}")

    def _update_dashboard_only(self):
        if not DASHBOARD_AVAILABLE: return
        try:
            now = datetime.now(timezone.utc)
            d_m, d_p = {}, {}
            for cid, m in self.markets.items():
                s, p = self.states.get(cid), self.positions.get(cid)
                if s:
                    d_m[cid] = {'asset': m.asset, 'prob': s.prob, 'time_left': (m.end_time-now).total_seconds()/60}
                    if p and p.size > 0:
                        cur = s.prob if p.side == "UP" else (1 - s.prob)
                        d_p[cid] = {'side': p.side, 'size': p.size, 'entry_price': p.entry_price, 'unrealized_pnl': (cur - p.entry_price) * (p.size/p.entry_price)}
            update_dashboard_state(strategy_name=self.strategy.name, total_pnl=self.total_pnl, trade_count=self.trade_count, win_count=self.win_count, positions=d_p, markets=d_m)
        except: pass

    async def decision_loop(self):
        while self.running:
            try:
                await asyncio.sleep(0.5)
                now_ts = time.time()
                
                if not self.markets or (now_ts - self.last_market_refresh > 30):
                    self.refresh_markets()
                    self.last_market_refresh = now_ts
                
                for cid, m in list(self.markets.items()):
                    state = self.states.get(cid)
                    if state is None: continue # Ochrana proti NoneType
                    
                    ob = self.orderbook_streamer.get_orderbook(cid, "UP")
                    if ob and ob.mid_price:
                        state.prob = ob.mid_price
                        self.execute_action(cid, state)
            except Exception as e: 
                logging.error(f"⚠️ Loop Error: {e}")

    async def run(self):
        self.running = True
        tasks = [
            asyncio.create_task(self.price_streamer.stream()),
            asyncio.create_task(self.orderbook_streamer.stream()),
            asyncio.create_task(self.futures_streamer.stream()),
            asyncio.create_task(self.decision_loop())
        ]
        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logging.info("Shutting down...")
        finally:
            self.running = False
            for t in tasks: t.cancel()
            if isinstance(self.strategy, RLStrategy) and getattr(self.strategy, 'training', False):
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
    if isinstance(strategy, RLStrategy) and args.train:
        strategy.train()
    
    engine = TradingEngine(strategy, trade_size=args.size)
    await engine.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
