import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === KONFIGURACE ===
INITIAL_CAPITAL = 1000
TRADE_SIZE = 500
SLIPPAGE = 0.005    # Sníženo na 0.5% (1% je pro HFT extrémní bariéra)
FIXED_PENALTY = 0.10 
ATR_WINDOW = 20     # Okno pro výpočet volatility
TP_MULT = 1.5       # Take Profit násobek ATR
SL_MULT = 1.0       # Stop Loss násobek ATR

class MasterSniperTPBacktester:
    def __init__(self, log_pattern="logs/market_ticks_hft_*.csv"):
        self.filename = self._find_latest_file(log_pattern)
        print(f"Analyzuji data: {self.filename}")
        self.df = pd.read_csv(self.filename)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['asset', 'timestamp'])

    def _find_latest_file(self, pattern):
        files = glob.glob(pattern)
        if not files: files = glob.glob("market_ticks_hft_*.csv")
        return max(files, key=os.path.getmtime)

    def run_simulation(self, threshold, tick_step=10, min_hold_sec=30, use_tp=True):
        total_balance = INITIAL_CAPITAL
        
        for asset, group in self.df.groupby('asset'):
            # Výpočet dynamické volatility (ATR-like) z pravděpodobnosti/ceny
            group = group.copy()
            group['atr'] = group['prob'].diff().abs().rolling(window=ATR_WINDOW).mean()
            
            pos = None
            sampled_group = group.iloc[::tick_step]
            prices = sampled_group['prob'].values
            atrs = sampled_group['atr'].values
            times = sampled_group['timestamp'].values
            
            for t in range(len(prices)):
                mid_price = prices[t]
                current_atr = atrs[t] if not np.isnan(atrs[t]) else 0.001
                current_time = times[t]
                
                if pos is None:
                    dist = abs(mid_price - 0.50)
                    if dist > (threshold - 0.50):
                        side = "UP" if mid_price > 0.50 else "DOWN"
                        entry_p = (mid_price if side == "UP" else (1 - mid_price)) * (1 + SLIPPAGE)
                        
                        # Definice dynamických hladin
                        tp_level = entry_p + (current_atr * TP_MULT)
                        sl_level = entry_p - (current_atr * SL_MULT)
                        
                        pos = {'entry_p': entry_p, 'entry_t': current_time, 
                               'side': side, 'tp': tp_level, 'sl': sl_level}
                
                else:
                    duration_sec = (current_time - pos['entry_t']) / np.timedelta64(1, 's')
                    curr_p_raw = mid_price if pos['side'] == "UP" else (1 - mid_price)
                    
                    # Logika výstupu: ATR Take Profit | ATR Stop Loss | Reversal
                    is_tp = use_tp and curr_p_raw >= pos['tp']
                    is_sl = use_tp and curr_p_raw <= pos['sl']
                    is_reversal = duration_sec >= min_hold_sec and (
                        (pos['side'] == "UP" and mid_price < 0.50) or 
                        (pos['side'] == "DOWN" and mid_price > 0.50)
                    )

                    if is_tp or is_sl or is_reversal:
                        exit_price = curr_p_raw * (1 - SLIPPAGE)
                        shares = TRADE_SIZE / pos['entry_p']
                        pnl = (exit_price - pos['entry_p']) * shares
                        
                        # Penalizace (asymetrická pro trénink)
                        pnl = (pnl - FIXED_PENALTY) if pnl > 0 else (pnl * 1.2 - FIXED_PENALTY)
                        
                        total_balance += pnl
                        pos = None
                            
        return total_balance - INITIAL_CAPITAL

# ... (zbytek main funkce zůstává podobný, jen volá novou logiku)
