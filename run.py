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

# Nastavení logování pro terminál
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
        self.markets, self.positions, self.states = {}, {}, {}
        
        # Buffery a filtry
        self.prob_buffers = {}    
        self.atr_buffers = {}     
        self.cooldowns = {}       
        self.prev_states = {}
        self.pending_rewards = {}
        
        self.total_pnl, self.trade_count, self.win_count = 0.0, 0, 0
        self.running = False
        
        # Ošetření timeoutů
        self.last_market_refresh = 0

    def execute_action(self, cid: str, state: MarketState):
        pos = self.positions.get(cid)
        if not pos: return
        now_ts = time.time()

        if cid not in self.prob_buffers:
            self.prob_buffers[cid] = deque(maxlen=10) 
            self.atr_buffers[cid] = deque(maxlen=30)

        # Ochrana: Pokud state.prob nepřišel, neprovádíme akci
        if state.prob is None or state.prob == 0: return

        self.prob_buffers[cid].append(state.prob)
        # Použijeme fixní volatilitu, pokud se nedaří načíst ATR z API, aby bot nezamrzl
        vol = state.realized_vol_5m if (hasattr(state, 'realized_vol_5m') and state.realized_vol_5m > 0) else 0.004
        self.atr_buffers[cid].append(vol)
        
        smoothed_prob = sum(self.prob_buffers[cid]) / len(self.prob_buffers[cid])
        avg_atr = sum(self.atr_buffers[cid]) / len(self.atr_buffers[cid])

        FEE, EXIT_FEE = 1.01, 0.99 

        # LOGIKA VÝSTUPU
        if pos.size > 0:
            curr_val = state.prob if pos.side == "UP" else (1 - state.prob)
            if curr_val >= pos.tp_level or curr_val <= pos.sl_level:
                shares = pos.size / pos.entry_price
                eff_exit = curr_val * EXIT_FEE
                pnl = (eff_exit - pos.entry_price) * shares
                self._record_trade(pos, eff_exit, pnl, f"CLOSE {pos.side}", cid=cid)
                pos.size, pos.side = 0, None
                self.cooldowns[cid] = now_ts + 10 # Krátký cooldown pro víc obchodů
            return

        # LOGIKA VSTUPU (REVERSAL)
        if pos.size == 0 and now_ts > self.cooldowns.get(cid, 0):
            # Sníženo na 3 ticky pro rychlejší reakci, aby udělal víc obchodů
            if len(self.prob_buffers[cid]) < 3: return

            # Agresivnější thresholdy pro více obchodů
            UPPER_THR = 0.60 
            LOWER_THR = 0.40

            if smoothed_prob > UPPER_THR:
                pos.side = "DOWN"
                pos.entry_price = (1 - state.prob) * FEE
                pos.tp_level = pos.entry_price + (avg_atr * 1.2)
                pos.sl_level = pos.entry_price - (avg_atr * 2.0)
                pos.size = self.trade_size
                pos.entry_time = datetime.now(timezone.utc)
                logging.info(f"📉 [REV] SHORT {pos.asset} @ {pos.entry_price:.3f}")

            elif smoothed_prob < LOWER_THR:
                pos.side = "UP"
                pos.entry_price = state.prob * FEE
                pos.tp_level = pos.entry_price + (avg_atr * 1.2)
                pos.sl_level = pos.entry_price - (avg_atr * 2.0)
                pos.size = self.trade_size
                pos.entry_time = datetime.now(timezone.utc)
                logging.info(f"📈 [REV] LONG {pos.asset} @ {pos.entry_price:.3f}")

    def _record_trade(self, pos: Position, price: float, pnl: float, action: str, cid: str = None):
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0: self.win_count += 1
        logging.info(f"💰 {action} {pos.asset} PnL: ${pnl:+.2f} | Total: ${self.total_pnl:.2f}")

    async def decision_loop(self):
        while self.running:
            try:
                # 0.5s je ideální pro stabilitu a rychlost
                await asyncio.sleep(0.5)
                now = datetime.now(timezone.utc)
                now_ts = time.time()
                
                # Refresh trhů jen jednou za 30 sekund, abychom šetřili REST API (zabrání Timeoutům)
                if not self.markets or (now_ts - self.last_market_refresh > 30):
                    try:
                        self.refresh_markets()
                        self.last_market_refresh = now_ts
                    except Exception as e:
                        logging.error(f"Market refresh failed (Binance Busy): {e}")
                        await asyncio.sleep(2); continue
                
                for cid, m in list(self.markets.items()):
                    if m.end_time <= now: continue
                    
                    state = self.states.get(cid)
                    if not state: continue

                    # Získáváme data z Orderbook Streameru (WebSocket = Žádný Timeout!)
                    ob = self.orderbook_streamer.get_orderbook(cid, "UP")
                    if ob and ob.mid_price:
                        state.prob = ob.mid_price
                        self.execute_action(cid, state)
                    
            except Exception as e:
                logging.error(f"Loop Error: {e}")

    def refresh_markets(self):
        # Tato funkce volá REST API, proto ji omezujeme časem výše
        markets = get_15m_markets(assets=["BTC", "ETH", "SOL", "XRP"])
        self.markets.clear()
        for m in markets:
            self.markets[m.condition_id] = m
            self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)
            if m.condition_id not in self.states:
                self.states[m.condition_id] = MarketState(asset=m.asset, prob=m.price_up)
            if m.condition_id not in self.positions:
                self.positions[m.condition_id] = Position(asset=m.asset)

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
        finally:
            self.running = False
            for t in tasks: t.cancel()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=AVAILABLE_STRATEGIES)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--size", type=float, default=10.0)
    args = parser.parse_args()

    strategy = create_strategy(args.strategy)
    engine = TradingEngine(strategy, trade_size=args.size)
    await engine.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
