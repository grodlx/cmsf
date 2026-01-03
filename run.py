#!/usr/bin/env python3
import asyncio
import argparse
import copy
import sys
import threading
import os
import time
import logging
import gc
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

try:
    from dashboard_cinematic import update_dashboard_state, update_rl_metrics, emit_rl_buffer, run_dashboard, emit_trade
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    def update_dashboard_state(**kwargs): pass
    def emit_trade(action, asset, size=0, pnl=None): pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

@dataclass
class Position:
    asset: str
    side: Optional[str] = None
    size: float = 0.0
    entry_price: float = 0.0
    entry_time: float = 0.0
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
        self.prob_buffers, self.atr_buffers, self.cooldowns = {}, {}, {}
        self.total_pnl, self.trade_count, self.win_count = 0.0, 0, 0
        self.running, self.last_cleanup, self.last_market_refresh = False, time.time(), 0

    def execute_action(self, cid: str, state: MarketState):
        pos = self.positions.get(cid)
        if not pos: return
        now_ts = time.time()

        if cid not in self.prob_buffers:
            self.prob_buffers[cid] = deque(maxlen=12) # Vyhlazování pro trend filtr
            self.atr_buffers[cid] = deque(maxlen=30)

        self.prob_buffers[cid].append(state.prob)
        vol = getattr(state, 'realized_vol_5m', 0.005)
        self.atr_buffers[cid].append(vol if vol > 0 else 0.005)
        
        smoothed_prob = sum(self.prob_buffers[cid]) / len(self.prob_buffers[cid])
        avg_atr = sum(self.atr_buffers[cid]) / len(self.atr_buffers[cid])

        FEE, EXIT_FEE = 1.01, 0.99 

        # --- LOGIKA VÝSTUPU ---
        if pos.size > 0:
            curr_val = state.prob if pos.side == "UP" else (1 - state.prob)
            duration = now_ts - pos.entry_time
            
            # Dynamické hladiny (TP blíž, SL dál pro "oddych")
            is_tp = curr_val >= pos.tp_level
            is_sl = curr_val <= pos.sl_level
            is_timeout = duration > 50 # Time exit

            if is_tp or is_sl or is_timeout:
                shares = pos.size / pos.entry_price
                eff_exit = curr_val * EXIT_FEE
                pnl = (eff_exit - pos.entry_price) * shares
                reason = "TP" if is_tp else ("SL" if is_sl else "TIME")
                self._record_trade(pos, eff_exit, pnl, f"CLOSE {pos.side} ({reason})", cid=cid)
                pos.size, pos.side = 0, None
                self.cooldowns[cid] = now_ts + 10 # Odpočinek po obchodu
            return

        # --- LOGIKA VSTUPU (S TRENDOVÝM FILTREM) ---
        if pos.size == 0 and now_ts > self.cooldowns.get(cid, 0):
            if len(self.prob_buffers[cid]) < 10: return

            # TREND FILTER: Pokud posledních 5 hodnot jde stále nahoru/dolů, nejdeme proti nim
            history = list(self.prob_buffers[cid])
            is_trending_up = all(history[i] < history[i+1] for i in range(-5, -1))
            is_trending_down = all(history[i] > history[i+1] for i in range(-5, -1))

            UPPER_THR = 0.63
            LOWER_THR = 0.37

            # SHORT REVERSAL (Pouze pokud cena neroste prudce)
            if smoothed_prob > UPPER_THR and not is_trending_up:
                pos.side, pos.entry_price = "DOWN", (1 - state.prob) * FEE
                # TP musí pokrýt 1% poplatek -> násobíme ATR
                pos.tp_level, pos.sl_level = pos.entry_price + (avg_atr * 1.3), pos.entry_price - (avg_atr * 2.5)
                pos.size, pos.entry_time = self.trade_size, now_ts
                logging.info(f"📉 [ANTI-TREND] SHORT {pos.asset} @ {pos.entry_price:.3f}")

            # LONG REVERSAL (Pouze pokud cena nepadá prudce)
            elif smoothed_prob < LOWER_THR and not is_trending_down:
                pos.side, pos.entry_price = "UP", state.prob * FEE
                pos.tp_level, pos.sl_level = pos.entry_price + (avg_atr * 1.3), pos.entry_price - (avg_atr * 2.5)
                pos.size, pos.entry_time = self.trade_size, now_ts
                logging.info(f"📈 [ANTI-TREND] LONG {pos.asset} @ {pos.entry_price:.3f}")

    def _record_trade(self, pos: Position, price: float, pnl: float, action: str, cid: str = None):
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0: self.win_count += 1
        logging.info(f"💰 {action} {pos.asset} | PnL: ${pnl:+.2f} | Realized Total: ${self.total_pnl:.2f}")
        emit_trade(action, pos.asset, pos.size, pnl)
        self._update_dashboard_only()

    def _update_dashboard_only(self):
        if not DASHBOARD_AVAILABLE: return
        try:
            update_dashboard_state(
                strategy_name=self.strategy.name,
                total_pnl=self.total_pnl,
                trade_count=self.trade_count,
                win_count=self.win_count,
                positions={cid: {'side': p.side, 'size': p.size, 'entry_price': p.entry_price} for cid, p in self.positions.items() if p.size > 0},
                markets={cid: {'asset': m.asset, 'prob': self.states[cid].prob} for cid, m in self.markets.items()}
            )
        except: pass

    async def decision_loop(self):
        while self.running:
            try:
                await asyncio.sleep(0.5)
                now_ts = time.time()
                if now_ts - self.last_cleanup > 1200:
                    gc.collect()
                    self.last_cleanup = now_ts
                if not self.markets or (now_ts - self.last_market_refresh > 40):
                    self.refresh_markets()
                    self.last_market_refresh = now_ts
                for cid, m in list(self.markets.items()):
                    state = self.states.get(cid)
                    if not state: continue
                    ob = self.orderbook_streamer.get_orderbook(cid, "UP")
                    if ob and ob.mid_price:
                        state.prob = ob.mid_price
                        self.execute_action(cid, state)
            except Exception as e: logging.error(f"Loop Error: {e}")

    def refresh_markets(self):
        try:
            markets = get_15m_markets(assets=["BTC", "ETH", "SOL", "XRP"])
            self.markets.clear()
            for m in markets:
                self.markets[m.condition_id] = m
                self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)
                if m.condition_id not in self.states:
                    self.states[m.condition_id] = MarketState(asset=m.asset, prob=m.price_up, time_remaining=1.0)
                if m.condition_id not in self.positions:
                    self.positions[m.condition_id] = Position(asset=m.asset)
        except Exception as e: logging.error(f"Refresh Error: {e}")

    async def run(self):
        self.running = True
        tasks = [asyncio.create_task(self.price_streamer.stream()), asyncio.create_task(self.orderbook_streamer.stream()),
                 asyncio.create_task(self.futures_streamer.stream()), asyncio.create_task(self.decision_loop())]
        try: await asyncio.gather(*tasks)
        finally:
            self.running = False
            for t in tasks: t.cancel()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=AVAILABLE_STRATEGIES)
    parser.add_argument("--size", type=float, default=10.0)
    parser.add_argument("--dashboard", action="store_true")
    args = parser.parse_args()
    if args.dashboard and DASHBOARD_AVAILABLE:
        threading.Thread(target=run_dashboard, kwargs={'port': 5051}, daemon=True).start()
    strategy = create_strategy(args.strategy)
    engine = TradingEngine(strategy, trade_size=args.size)
    await engine.run()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
