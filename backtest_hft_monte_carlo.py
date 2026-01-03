import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

# === KONFIGURACE ===
INITIAL_CAPITAL = 1000
TRADE_SIZE = 500
SLIPPAGE = 0.01   # 1%
FIXED_PENALTY = 0.15 

class PatientSniperBacktester:
    def __init__(self, log_pattern="logs/market_ticks_hft_*.csv"):
        self.filename = self._find_latest_file(log_pattern)
        print(f"Načítám data: {self.filename}")
        self.df = pd.read_csv(self.filename)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['asset', 'timestamp'])

    def _find_latest_file(self, pattern):
        files = glob.glob(pattern)
        if not files: files = glob.glob("market_ticks_hft_*.csv")
        return max(files, key=os.path.getmtime)

    def run_simulation(self, threshold, min_hold_sec=30, take_profit=0.03):
        total_balance = INITIAL_CAPITAL
        
        for asset, group in self.df.groupby('asset'):
            pos = None
            prices = group['prob'].values
            times = group['timestamp'].values
            
            for t in range(len(prices)):
                mid_price = prices[t]
                current_time = times[t]
                
                # LOGIKA VSTUPU
                if pos is None:
                    dist = abs(mid_price - 0.50)
                    if dist > (threshold - 0.50):
                        side = "UP" if mid_price > 0.50 else "DOWN"
                        # Vstupní cena + 1% slippage
                        entry_price = (mid_price if side == "UP" else (1 - mid_price)) * (1 + SLIPPAGE)
                        pos = {'entry_p': entry_price, 'entry_t': current_time, 'side': side}
                
                # LOGIKA VÝSTUPU
                else:
                    duration_sec = (current_time - pos['entry_t']) / np.timedelta64(1, 's')
                    
                    # Aktuální čistá cena při prodeji (zahrnuje 1% slippage na výstupu)
                    curr_exit_raw = mid_price if pos['side'] == "UP" else (1 - mid_price)
                    curr_exit_p = curr_exit_raw * (1 - SLIPPAGE)
                    
                    shares = TRADE_SIZE / pos['entry_p']
                    current_pnl = (curr_exit_p - pos['entry_p']) * shares
                    
                    # PATIENT LOGIKA: 
                    # 1. Take Profit: Pokud jsme v zisku aspoň X procent (např 3%), bereme ho.
                    # 2. Reversal: Pokud signál zmizel A jsme v pozici aspoň min_hold_sec.
                    
                    is_take_profit = current_pnl >= (TRADE_SIZE * take_profit)
                    is_trend_over = duration_sec >= min_hold_sec and (
                        (pos['side'] == "UP" and mid_price < 0.50) or 
                        (pos['side'] == "DOWN" and mid_price > 0.50)
                    )

                    if is_take_profit or is_trend_over:
                        total_balance += current_pnl
                        pos = None
                            
        return total_balance - INITIAL_CAPITAL

def main():
    tester = PatientSniperBacktester()

    # TESTUJEME PACIENTNÍ SNIPERY
    scenarios = {
        "Old: Aggressive (0.56)": (0.56, 30, 0.02),
        "Patient: Thr 0.58 + TP 3%": (0.58, 60, 0.03),
        "Patient: Thr 0.60 + TP 4%": (0.60, 120, 0.04),
        "Extreme: Thr 0.62 + TP 5%": (0.62, 300, 0.05)
    }
    
    final_results = {}
    print("\n" + "="*60)
    print("MONTE CARLO: PATIENT SNIPER MODE")
    print("="*60)

    for name, params in scenarios.items():
        data = [tester.run_simulation(params[0], params[1], params[2]) for _ in range(20)]
        final_results[name] = data
        print(f"{name:25} | Mean PnL: ${np.mean(data):8.2f} | Win: {sum(1 for r in data if r > 0)/len(data)*100:5.1f}%")

    plt.figure(figsize=(12, 7))
    plt.boxplot(final_results.values(), labels=final_results.keys())
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.title("Patient Sniper: Výsledek s korektním ošetřením poplatků")
    plt.show()

if __name__ == "__main__":
    main()
