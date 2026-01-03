import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

# === KONFIGURACE ===
INITIAL_CAPITAL = 1000
TRADE_SIZE = 500
SLIPPAGE = 0.01   # 1% na vstupu i výstupu
FIXED_PENALTY = 0.15 

class MasterSniperNoTPBacktester:
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

    def run_simulation(self, threshold, tick_step=10, min_hold_sec=60, penalty=True):
        total_balance = INITIAL_CAPITAL
        
        for asset, group in self.df.groupby('asset'):
            pos = None
            # Sampling: simulace intervalu (např. 10 = 5s na 0.5s datech)
            sampled_group = group.iloc[::tick_step]
            prices = sampled_group['prob'].values
            times = sampled_group['timestamp'].values
            
            for t in range(len(prices)):
                mid_price = prices[t]
                current_time = times[t]
                
                if pos is None:
                    dist = abs(mid_price - 0.50)
                    if dist > (threshold - 0.50):
                        side = "UP" if mid_price > 0.50 else "DOWN"
                        # Vstup + 1% slippage
                        entry_price = (mid_price if side == "UP" else (1 - mid_price)) * (1 + SLIPPAGE)
                        pos = {'entry_p': entry_price, 'entry_t': current_time, 'side': side}
                
                else:
                    duration_sec = (current_time - pos['entry_t']) / np.timedelta64(1, 's')
                    
                    # LOGIKA VÝSTUPU (Pouze Reversal po uplynutí min_hold_sec)
                    if duration_sec >= min_hold_sec:
                        # Pokud signál padne pod 0.50 u UP nebo nad 0.50 u DOWN
                        should_close = (pos['side'] == "UP" and mid_price < 0.50) or \
                                       (pos['side'] == "DOWN" and mid_price > 0.50)
                        
                        if should_close:
                            exit_p_raw = mid_price if pos['side'] == "UP" else (1 - mid_price)
                            exit_price = exit_p_raw * (1 - SLIPPAGE) # Výstup - 1% slippage
                            
                            shares = TRADE_SIZE / pos['entry_p']
                            final_pnl = (exit_price - pos['entry_p']) * shares
                            
                            if penalty:
                                # Asymetrická penalizace pro RL
                                final_pnl = (final_pnl - FIXED_PENALTY) if final_pnl > 0 else (final_pnl * 1.5 - FIXED_PENALTY)
                            
                            total_balance += final_pnl
                            pos = None
                            
        return total_balance - INITIAL_CAPITAL

def main():
    tester = MasterSniperNoTPBacktester()

    scenarios = {
        "HFT (0.5s, Thr 0.55)": (0.55, 1, 30, False),
        "Sniper 5s (Thr 0.60)": (0.60, 10, 60, False),
        "Master Sniper 5s (No TP)": (0.62, 10, 120, True) # 5s tick, 120s hold, Penalty
    }
    
    final_results = {}
    print("\n" + "="*60)
    print("VERIFIKACE: MASTER SNIPER 5s (BEZ TAKE PROFIT)")
    print("="*60)

    for name, params in scenarios.items():
        data = [tester.run_simulation(*params) for _ in range(30)]
        final_results[name] = data
        print(f"{name:25} | Mean PnL: ${np.mean(data):8.2f} | Win: {sum(1 for r in data if r > 0)/len(data)*100:5.1f}%")

    plt.figure(figsize=(12, 7))
    plt.boxplot(final_results.values(), labels=final_results.keys())
    plt.axhline(0, color='green', linestyle='-', alpha=0.3)
    plt.title("Master Sniper (No TP) vs HFT")
    plt.show()

if __name__ == "__main__":
    main()
