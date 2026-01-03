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
        self.df = pd.read_csv(self.filename)
        # Převod na datetime pro správné řazení
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['asset', 'timestamp'])
        print(f"Načtena HFT data: {self.filename} ({len(self.df)} tiků)")

    def _find_latest_file(self, pattern):
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError("Nebyl nalezen žádný soubor market_ticks_hft_*.csv!")
        return max(files, key=os.path.getctime)

    def run_simulation(self, cooldown_sec, threshold, penalty_mode=False):
        """Simuluje portfolio obchodování na všech aktivech v logu."""
        total_balance = INITIAL_CAPITAL
        # Sledujeme pozice pro každé aktivum zvlášť
        active_positions = {} # {asset: {'entry_p': price, 'entry_t': time, 'side': side, 'size': size}}
        
        # Seskupíme data podle aktiv, abychom simulovali každé v čase
        for asset, group in self.df.groupby('asset'):
            pos = None
            prices = group['prob'].values
            times = group['timestamp'].values
            
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
                    duration = (current_time - pos['entry_t']).total_seconds()
                    
                    # Kontrola Cooldownu a otočení signálu
                    if duration >= cooldown_sec:
                        should_close = (pos['side'] == "UP" and mid_price < 0.49) or \
                                       (pos['side'] == "DOWN" and mid_price > 0.51)
                        
                        if should_close:
                            exit_p_raw = mid_price if pos['side'] == "UP" else (1 - mid_price)
                            exit_price = exit_p_raw * (1 - SLIPPAGE)
                            
                            shares = TRADE_SIZE / pos['entry_p']
                            trade_pnl = (exit_price - pos['entry_p']) * shares
                            
                            # Penalizace pro Ultra model (učení agenta)
                            if penalty_mode:
                                trade_pnl = (trade_pnl - FIXED_PENALTY) if trade_pnl > 0 else (trade_pnl * 1.5 - FIXED_PENALTY)
                            
                            total_balance += trade_pnl
                            pos = None # Pozice uzavřena
                            
        return total_balance - INITIAL_CAPITAL

def main():
    try:
        tester = HFTBacktester()
    except Exception as e:
        print(f"Chyba: {e}")
        return

    # Definice scénářů pro testování
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
        # Monte Carlo prvek: Spustíme simulaci s mírnými náhodnými odchylkami v thresholdu
        # To simuluje realitu, kde exekuce není vždy v milisekundě přesná
        data = []
        for _ in range(30): # 30 iterací pro stabilitu
            noise = np.random.uniform(-0.003, 0.003)
            res = tester.run_simulation(params[0], params[1] + noise, params[2])
            data.append(res)
        
        final_results[name] = data
        mean_pnl = np.mean(data)
        win_rate = sum(1 for r in data if r > 0) / len(data) * 100
        print(f"{name:25} | PnL: ${mean_pnl:8.2f} | Win: {win_rate:5.1f}%")

    # Grafické srovnání
    plt.figure(figsize=(12, 7))
    plt.boxplot(final_results.values(), labels=final_results.keys())
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f"HFT Monte Carlo Backtest (Data: {os.path.basename(tester.filename)})")
    plt.ylabel("Čistý zisk/ztráta ($)")
    plt.grid(True, axis='y', alpha=0.3)
    
    output_img = "hft_backtest_result.png"
    plt.savefig(output_img)
    print(f"\nGraf byl uložen jako '{output_img}'")
    plt.show()

if __name__ == "__main__":
    main()
