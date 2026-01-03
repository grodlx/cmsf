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
STOP_LOSS_USD = 7.5 # Stop loss nastaven na 1.5% z TRADE_SIZE (500 * 0.015)

class HFTSniperBacktester:
    def __init__(self, log_pattern="logs/market_ticks_hft_*.csv"):
        self.filename = self._find_latest_file(log_pattern)
        print(f"Načítám HFT data: {self.filename}")
        self.df = pd.read_csv(self.filename)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['asset', 'timestamp'])

    def _find_latest_file(self, pattern):
        files = glob.glob(pattern)
        if not files:
            files = glob.glob("market_ticks_hft_*.csv")
        if not files:
            raise FileNotFoundError("Data nebyla nalezena!")
        return max(files, key=os.path.getmtime)

    def run_simulation(self, cooldown_sec, threshold, use_sl=False, penalty_mode=False):
        total_balance = INITIAL_CAPITAL
        
        for asset, group in self.df.groupby('asset'):
            pos = None
            prices = group['prob'].values
            times = group['timestamp'].values
            
            for t in range(len(prices)):
                mid_price = prices[t]
                current_time = times[t]
                
                # LOGIKA VSTUPU (SNIPER)
                if pos is None:
                    dist = abs(mid_price - 0.50)
                    if dist > (threshold - 0.50):
                        side = "UP" if mid_price > 0.50 else "DOWN"
                        entry_price = (mid_price if side == "UP" else (1 - mid_price)) * (1 + SLIPPAGE)
                        pos = {'entry_p': entry_price, 'entry_t': current_time, 'side': side}
                
                # LOGIKA VÝSTUPU
                else:
                    duration_sec = (current_time - pos['entry_t']) / np.timedelta64(1, 's')
                    
                    # Aktuální cena pro exit (pro účely Stop-Lossu)
                    curr_exit_raw = mid_price if pos['side'] == "UP" else (1 - mid_price)
                    curr_exit_p = curr_exit_raw * (1 - SLIPPAGE)
                    shares = TRADE_SIZE / pos['entry_p']
                    current_pnl = (curr_exit_p - pos['entry_p']) * shares
                    
                    # Podmínky pro zavření
                    is_stop_loss = use_sl and current_pnl <= -STOP_LOSS_USD
                    is_signal_reversal = duration_sec >= cooldown_sec and (
                        (pos['side'] == "UP" and mid_price < 0.50) or 
                        (pos['side'] == "DOWN" and mid_price > 0.50)
                    )

                    if is_stop_loss or is_signal_reversal:
                        trade_pnl = current_pnl
                        if penalty_mode:
                            trade_pnl = (trade_pnl - FIXED_PENALTY) if trade_pnl > 0 else (trade_pnl * 1.5 - FIXED_PENALTY)
                        
                        total_balance += trade_pnl
                        pos = None
                            
        return total_balance - INITIAL_CAPITAL

def main():
    tester = HFTSniperBacktester()

    # NOVÉ STRATEGIE K OVĚŘENÍ
    scenarios = {
        "Old: Aggressive (0.56)": (45, 0.56, False, False),
        "Sniper: Thr 0.60": (10, 0.60, False, False),
        "Sniper: Thr 0.63 + SL": (0, 0.63, True, False), # SL = Stop Loss
        "Sniper: Thr 0.65 + SL": (0, 0.65, True, True)   # Nejpřísnější model
    }
    
    final_results = {}
    print("\n" + "="*60)
    print("MONTE CARLO: SNIPER MODE VERIFICATION")
    print("="*60)

    for name, params in scenarios.items():
        data = []
        for _ in range(30):
            # Přidáváme šum, abychom viděli, co udělá malá změna ceny na Polymarketu
            noise = np.random.uniform(-0.005, 0.005)
            res = tester.run_simulation(params[0], params[1] + noise, params[2], params[3])
            data.append(res)
        
        final_results[name] = data
        print(f"{name:25} | Mean PnL: ${np.mean(data):8.2f} | Win: {sum(1 for r in data if r > 0)/len(data)*100:5.1f}%")

    plt.figure(figsize=(12, 7))
    plt.boxplot(final_results.values(), labels=final_results.keys())
    plt.axhline(0, color='green', linestyle='-', alpha=0.3)
    plt.title(f"Sniper Backtest: {os.path.basename(tester.filename)}")
    plt.ylabel("Zisk / Ztráta ($)")
    plt.show()

if __name__ == "__main__":
    main()
