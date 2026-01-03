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
        self.prob_buffers = {}    # Pro vyhlazení signálu (SMA)
        self.atr_buffers = {}     # Pro stabilní výpočet volatility
        self.cooldowns = {}       # Ochrana proti overtradingu
        
        self.total_pnl, self.trade_count, self.win_count = 0.0, 0, 0
        self.running = False
        
        os.makedirs("logs", exist_ok=True)
        self.tick_log_path = f"logs/market_ticks_hft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._init_tick_log()

    def _init_tick_log(self):
        with open(self.tick_log_path, "w") as f:
            f.write("timestamp,asset,prob,binance_price,vol_5m,spread\n")

    def execute_action(self, cid: str, state: MarketState):
        pos = self.positions.get(cid)
        if not pos: return
        now_ts = time.time()

        # 1. --- FILTRACE SIGNÁLU (Smoothing) ---
        if cid not in self.prob_buffers:
            self.prob_buffers[cid] = deque(maxlen=5) # 5 ticků = cca 2.5s vyhlazení
            self.atr_buffers[cid] = deque(maxlen=20)

        self.prob_buffers[cid].append(state.prob)
        self.atr_buffers[cid].append(state.realized_vol_5m if state.realized_vol_5m > 0 else 0.002)
        
        smoothed_prob = sum(self.prob_buffers[cid]) / len(self.prob_buffers[cid])
        avg_atr = sum(self.atr_buffers[cid]) / len(self.atr_buffers[cid])

        FEE, EXIT_FEE = 1.01, 0.99 # Tvůj požadavek na 1% poplatek

        # 2. --- LOGIKA VÝSTUPU (Hlídání běžící pozice) ---
        if pos.size > 0:
            # U Reversalu měříme zisk/ztrátu vůči směru
            curr_val = state.prob if pos.side == "UP" else (1 - state.prob)
            
            is_tp = curr_val >= pos.tp_level
            is_sl = curr_val <= pos.sl_level
            
            if is_tp or is_sl:
                shares = pos.size / pos.entry_price
                eff_exit = curr_val * EXIT_FEE
                pnl = (eff_exit - pos.entry_price) * shares
                
                reason = "TAKE_PROFIT" if is_tp else "STOP_LOSS"
                self._record_trade(pos, eff_exit, pnl, f"CLOSE {pos.side} ({reason})", cid=cid)
                
                pos.size, pos.side = 0, None
                self.cooldowns[cid] = now_ts + 10 # 10s cooldown po uzavření
            return

        # 3. --- LOGIKA VSTUPU (REVERSAL s filtrem) ---
        # Vstupujeme jen pokud je signál vyhlazený a nejsme v cooldownu
        if pos.size == 0 and now_ts > self.cooldowns.get(cid, 0):
            # Musí být dostatek dat v bufferu pro stabilitu
            if len(self.prob_buffers[cid]) < 5: return

            UPPER_THR = 0.62
            LOWER_THR = 0.38

            # REVERSAL: Prob > 0.62 -> SHORT (sázka na návrat k 0.5)
            if smoothed_prob > UPPER_THR:
                pos.side = "DOWN"
                pos.entry_price = (1 - state.prob) * FEE # Vstup za reálnou cenu + fee
                # Nastavení hladin (TP blíž kvůli 1% fee, SL dál pro prostor)
                pos.tp_level = pos.entry_price + (avg_atr * 1.5)
                pos.sl_level = pos.entry_price - (avg_atr * 2.5)
                
                pos.size = self.trade_size
                pos.entry_time = datetime.now(timezone.utc)
                print(f"📉 [REV-ENTRY] SHORT {pos.asset} @ {pos.entry_price:.3f} (S-Prob: {smoothed_prob:.3f})")

            # REVERSAL: Prob < 0.38 -> LONG (sázka na návrat k 0.5)
            elif smoothed_prob < LOWER_THR:
                pos.side = "UP"
                pos.entry_price = state.prob * FEE
                pos.tp_level = pos.entry_price + (avg_atr * 1.5)
                pos.sl_level = pos.entry_price - (avg_atr * 2.5)
                
                pos.size = self.trade_size
                pos.entry_time = datetime.now(timezone.utc)
                print(f"📈 [REV-ENTRY] LONG {pos.asset} @ {pos.entry_price:.3f} (S-Prob: {smoothed_prob:.3f})")

    def _record_trade(self, pos: Position, price: float, pnl: float, action: str, cid: str = None):
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0: self.win_count += 1
        print(f"    {action} {pos.asset} @ {price:.3f} | Realized PnL: ${pnl:+.2f}")
        emit_trade(action, pos.asset, pos.size, pnl)
        self._update_dashboard_only()

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
        tick, tick_interval = 0, 0.5 # 500ms reakce
        while self.running:
            try:
                await asyncio.sleep(tick_interval)
                tick += 1
                now = datetime.now(timezone.utc)
                
                if not self.markets:
                    self.refresh_markets()
                    await asyncio.sleep(5); continue
                    
                for cid, m in self.markets.items():
                    if m.end_time <= now: continue
                    
                    state = self.states.get(cid)
                    ob = self.orderbook_streamer.get_orderbook(cid, "UP")
                    if ob and ob.mid_price: 
                        state.prob = ob.mid_price
                        state.spread = ob.spread or 0.0
                    
                    # Spuštění Reversal Engine
                    self.execute_action(cid, state)
                    
                if tick % 40 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] PnL: ${self.total_pnl:+.2f} | Trades: {self.trade_count} | WinRate: { (self.win_count/self.trade_count*100) if self.trade_count > 0 else 0 :.1f}%")
                    self._update_dashboard_only()
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
            print("\nShutting down...")
        finally:
            self.running = False
            for t in tasks: t.cancel()
            self.close_all_positions()

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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
