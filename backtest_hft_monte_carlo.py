import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

# === KONFIGURACE ===
INITIAL_CAPITAL = 1000
TRADE_SIZE = 500  # Velikost jedné pozice v USD
SLIPPAGE = 0.01   # 1% (poplatek při nákupu i prodeji)
FIXED_PENALTY = 0.15 # Fixní náklad pro Ultra model

class HFTBacktester:
    def __init__(self, log_pattern="logs/market_ticks_hft_*.csv"):
        self.filename = self._find_latest_file(log_pattern)
        print(f"Pokouším se načíst: {self.filename}")
        
        self.df = pd.read_csv(self.filename)
        # Převod na datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['asset', 'timestamp'])
        print(f"Úspěšně načteno: {len(self.df)} tiků.")

    def _find_latest_file(self, pattern):
        # Najde všechny soubory odpovídající masce
        files = glob.glob(pattern)
        if not files:
            # Záchranný pokus v aktuálním adresáři
            files = glob.glob("market_ticks_hft_*.csv")
            
        if not files:
            raise FileNotFoundError("Nebyl nalezen žádný soubor 'market_ticks_hft_*.csv'!")
        
        # Vrátí soubor s nejnovějším časem změny
        return max(files, key=os.path.getmtime)

    def run_simulation(self, cooldown_sec, threshold, penalty_mode=False):
        """Simuluje portfolio obchodování na všech aktivech v logu."""
        total_balance = INITIAL_CAPITAL
        
        for asset, group in self.df.groupby('asset'):
            pos = None
            prices = group['prob'].values
            times = group['timestamp'].values # numpy datetime64 array
            
            for t in range(len(prices)):
                mid_price = prices[t]
                current_time = times[t]
                signal_strength = abs(mid_price - 0.50)
                
                # LOGIKA VSTUPU
                if pos is None:
                    if signal_strength > (threshold - 0.50):
                        side = "UP" if mid_price > 0.50 else "DOWN"
                        # Cena s 1% slippage
                        entry_price = (mid_price if side == "UP" else (1 - mid_price)) * (1 + SLIPPAGE)
                        pos = {
                            'entry_p': entry_price,
                            'entry_t': current_time,
                            'side': side
                        }
                
                # LOGIKA VÝSTUPU
                else:
                    # OPRAVA: Výpočet trvání pozice pro numpy datetime64
                    duration_sec = (current_time - pos['entry_t']) / np.timedelta64(1, 's')
                    
                    if duration_sec >= cooldown_sec:
                        # Podmínka uzavření (při otočení trendu zpět k 0.50)
                        should_close = (pos['side'] == "UP" and mid_price < 0.495) or \
                                       (pos['side'] == "DOWN" and mid_price > 0.505)
                        
                        if should_close:
                            exit_p_raw = mid_price if pos['side'] == "UP" else (1 - mid_price)
                            exit_price = exit_p_raw * (1 - SLIPPAGE)
                            
                            shares = TRADE_SIZE / pos['entry_p']
                            trade_pnl = (exit_price - pos['entry_p']) * shares
                            
                            if penalty_mode:
                                # Asymetrická penalizace pro učení agenta
                                trade_pnl = (trade_pnl - FIXED_PENALTY) if trade_pnl > 0 else (trade_pnl * 1.5 - FIXED_PENALTY)
                            
                            total_balance += trade_pnl
                            pos = None
                            
        return total_balance - INITIAL_CAPITAL

def main():
    try:
        tester = HFTBacktester()
    except Exception as e:
        print(f"Chyba: {e}")
        return

    scenarios = {
        "HFT: Baseline (No Filters)": (0, 0.50, False),
        "HFT: Cooldown (60s)": (60, 0.50, False),
        "HFT: Aggressive (Thr 0.56)": (45, 0.56, False),
        "HFT: Ultra (Thr 0.57 + Pen)": (90, 0.57, True)
    }
    
    final_results = {}
    print("\n" + "="*50)
    print("VÝSLEDKY MONTE CARLO (HFT TICK DATA)")
    print("="*50)

    for name, params in scenarios.items():
        data = []
        for _ in range(30):
            noise = np.random.uniform(-0.002, 0.002)
            res = tester.run_simulation(params[0], params[1] + noise, params[2])
            data.append(res)
        
        final_results[name] = data
        mean_pnl = np.mean(data)
        win_rate = sum(1 for r in data if r > 0) / len(data) * 100
        print(f"{name:25} | PnL: ${mean_pnl:8.2f} | Win: {win_rate:5.1f}%")

    plt.figure(figsize=(12, 7))
    plt.boxplot(final_results.values(), labels=final_results.keys())
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f"HFT Monte Carlo Backtest (Data: {os.path.basename(tester.filename)})")
    plt.ylabel("Zisk / Ztráta ($)")
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.savefig("hft_monte_carlo_final.png")
    print(f"\nGraf uložen jako 'hft_monte_carlo_final.png'")
    plt.show()

if __name__ == "__main__":
    main()
