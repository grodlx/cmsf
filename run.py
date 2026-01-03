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

# Dashboard integrace
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
        
        # Buffery s omezenou délkou (Bod 1 & 4)
        self.prob_buffers = {}    # Pro vyhlazení (krátké)
        self.mean_buffers = {}    # Pro dynamický střed (dlouhé - Bod 3)
        self.atr_buffers = {}     
        self.cooldowns = {}       
        
        self.total_pnl, self.trade_count, self.win_count = 0.0, 0, 0
        self.running = False
        self.last_market_refresh = 0
        self.last_cleanup = time.time()

    def execute_action(self, cid: str, state: MarketState):
        pos = self.positions.get(cid)
        if not pos: return
        now_ts = time.time()

        # Inicializace bufferů pokud neexistují
        if cid not in self.prob_buffers:
            self.prob_buffers[cid] = deque(maxlen=10) # 5s vyhlazení
            self.mean_buffers[cid] = deque(maxlen=200) # ~10 min pro určení "středu"
            self.atr_buffers[cid] = deque(maxlen=30)

        # Ukládání dat
        self.prob_buffers[cid].append(state.prob)
        self.mean_buffers[cid].append(state.prob)
        
        vol = getattr(state, 'realized_vol_5m', 0.005)
        if vol <= 0: vol = 0.005
        self.atr_buffers[cid].append(vol)
        
        # Výpočty (Bod 3 - Dynamika)
        smoothed_prob = sum(self.prob_buffers[cid]) / len(self.prob_buffers[cid])
        current_mean = sum(self.mean_buffers[cid]) / len(self.mean_buffers[cid])
        avg_atr = sum(self.atr_buffers[cid]) / len(self.atr_buffers[cid])

        FEE, EXIT_FEE = 1.01, 0.99 

        # Hlídání otevřené pozice
        if pos.size > 0:
            curr_val = state.prob if pos.side == "UP" else (1 - state.prob)
            if curr_val >= pos.tp_level or curr_val <= pos.sl_level:
                shares = pos.size / pos.entry_price
                eff_exit = curr_val * EXIT_FEE
                pnl = (eff_exit - pos.entry_price) * shares
                self._record_trade(pos, eff_exit, pnl, f"CLOSE {pos.side}", cid=cid)
                pos.size, pos.side = 0, None
                self.cooldowns[cid] = now_ts + 10
            return

        # Vstupní logika (Reversal vůči dynamickému středu)
        if pos.size == 0 and now_ts > self.cooldowns.get(cid, 0):
            if len(self.mean_buffers[cid]) < 50: return # Potřebujeme data pro střed

            # Bod 3: Thresholdy jsou relativní k průměru posledních 10 minut
            # Pokud je průměr 0.55, tak Horní hranice není 0.60, ale 0.55 + 0.10 = 0.65
            UPPER_THR = current_mean + 0.08
            LOWER_THR = current_mean - 0.08

            # Ochrana: Thresholdy nesmí vyletět mimo logické hranice
            UPPER_THR = min(max(UPPER_THR, 0.58), 0.75)
            LOWER_THR = max(min(LOWER_THR, 0.42), 0.25)

            if smoothed_prob > UPPER_THR:
                pos.side = "DOWN"
                pos.entry_price = (1 - state.prob) * FEE
                pos.tp_level = pos.entry_price + (avg_atr * 1.2)
                pos.sl_level = pos.entry_price - (avg_atr * 2.5)
                pos.size = self.trade_size
                logging.info(f"📉 [DYNAMIC-REV] SHORT {pos.asset} | Prob: {smoothed_prob:.3f} | Mean: {current_mean:.3f}")

            elif smoothed_prob < LOWER_THR:
                pos.side = "UP"
                pos.entry_price = state.prob * FEE
                pos.tp_level = pos.entry_price + (avg_atr * 1.2)
                pos.sl_level = pos.entry_price - (avg_atr * 2.5)
                pos.size = self.trade_size
                logging.info(f"📈 [DYNAMIC-REV] LONG {pos.asset} | Prob: {smoothed_prob:.3f} | Mean: {current_mean:.3f}")

    def _record_trade(self, pos: Position, price: float, pnl: float, action: str, cid: str = None):
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0: self.win_count += 1
        logging.info(f"💰 {action} {pos.asset} PnL: ${pnl:+.2f} | Total: ${self.total_pnl:.2f}")
        emit_trade(action, pos.asset, pos.size, pnl)

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
                if m.condition_id not in self.states:
                    self.states[m.condition_id] = MarketState(asset=m.asset, prob=m.price_up, time_remaining=mins_left/15.0)
                if m.condition_id not in self.positions:
                    self.positions[m.condition_id] = Position(asset=m.asset)
        except Exception as e: logging.error(f"Refresh Error: {e}")

    async def decision_loop(self):
        while self.running:
            try:
                await asyncio.sleep(0.5)
                now_ts = time.time()
                
                # Bod 1 & 4: Pravidelné čištění paměti a reset driftu každých 30 minut
                if now_ts - self.last_cleanup > 1800:
                    logging.info("🧹 Periodic system cleanup (Memory & Drift reset)...")
                    gc.collect() # Vynucené uvolnění RAM
                    self.prob_buffers.clear() # Resetujeme krátké buffery pro novou kalibraci
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
                        # Ochrana proti šílenému spreadu (Bod 4)
                        if ob.spread and ob.spread > 0.01: continue 
                        self.execute_action(cid, state)
            except Exception as e: logging.error(f"Loop Error: {e}")

    async def run(self):
        self.running = True
        tasks = [
            asyncio.create_task(self.price_streamer.stream()),
            asyncio.create_task(self.orderbook_streamer.stream()),
            asyncio.create_task(self.futures_streamer.stream()),
            asyncio.create_task(self.decision_loop())
        ]
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
        threading.Thread(target=run_dashboard, kwargs={'port': 5050}, daemon=True).start()

    strategy = create_strategy(args.strategy)
    engine = TradingEngine(strategy, trade_size=args.size)
    await engine.run()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
