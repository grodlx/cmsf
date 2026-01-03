import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

# === KONFIGURACE ===
INITIAL_CAPITAL = 1000
TRADE_SIZE = 500
SLIPPAGE = 0.01  # 1% (vaše nastavení v run.py)
FIXED_PENALTY = 0.15 # Fixní náklad na obchod pro Ultra-Aggressive model

class RealDataBacktester:
    def __init__(self, log_pattern="logs/*.csv"):
        self.data = self._load_data(log_pattern)
        
    def _load_data(self, pattern):
        files = glob.glob(pattern)
        if not files:
            # Pokud nejsou data, vytvoříme dummy data pro ukázku
            print("WARN: Nebyla nalezena realná data v logs/. Generuji syntetický vzorek.")
            return pd.DataFrame({'prob_at_exit': np.random.uniform(0.45, 0.55, 1000)})
        
        all_df = []
        for f in files:
            try:
                df = pd.read_csv(f)
                all_df.append(df)
            except Exception as e:
                print(f"Chyba při načítání {f}: {e}")
        
        combined = pd.concat(all_df)
        print(f"Úspěšně načteno {len(combined)} řádků z realných logů.")
        return combined

    def run_scenario(self, cooldown, threshold, penalty_mode=False):
        balance = INITIAL_CAPITAL
        in_pos = False
        entry_p, entry_t, side = 0, 0, None
        
        # Simulujeme průběh trhu z CSV
        # Pokud log obsahuje 'prob_at_exit', použijeme ho jako Mid-Price
        prices = self.data['prob_at_exit'].values
        
        for t, mid_price in enumerate(prices):
            signal_strength = abs(mid_price - 0.50)
            
            # LOGIKA VSTUPU
            if not in_pos:
                # Threshold filtr
                if signal_strength > (threshold - 0.50):
                    in_pos = True
                    entry_t = t
                    side = "UP" if mid_price > 0.50 else "DOWN"
                    
                    # REALITA NÁKUPU: 
                    # Kupujeme za cenu horší o Slippage (simulace spreadu + poplatku)
                    actual_prob = mid_price if side == "UP" else (1 - mid_price)
                    entry_p = actual_prob * (1 + SLIPPAGE)
            
            # LOGIKA VÝSTUPU
            elif in_pos:
                # Cooldown filtr (počet tiků v CSV)
                if (t - entry_t) >= cooldown:
                    # Podmínka pro zavření: signál se otočil nebo vyprchal
                    should_close = (side == "UP" and mid_price < 0.49) or (side == "DOWN" and mid_price > 0.51)
                    
                    if should_close:
                        # REALITA PRODEJE: 
                        # Prodáváme za cenu horší o Slippage
                        exit_p_raw = mid_price if side == "UP" else (1 - mid_price)
                        exit_p = exit_p_raw * (1 - SLIPPAGE)
                        
                        shares = TRADE_SIZE / entry_p
                        trade_pnl = (exit_p - entry_p) * shares
                        
                        # ASYMETRICKÁ PENALIZACE (z vašeho nového modelu)
                        if penalty_mode:
                            # Trestáme ztráty víc (1.5x) a odečítáme fixní náklad
                            trade_pnl = (trade_pnl - FIXED_PENALTY) if trade_pnl > 0 else (trade_pnl * 1.5 - FIXED_PENALTY)
                        
                        balance += trade_pnl
                        in_pos = False
        
        return balance - INITIAL_CAPITAL

def main():
    tester = RealDataBacktester()

    # Definice testovaných strategií
    scenarios = {
        "Real: No Filters": (0, 0.50, False),
        "Real: Cooldown 30": (30, 0.50, False),
        "Real: Ultra-Aggressive": (45, 0.56, True)
    }
    
    final_stats = {}
    
    # Pro vizuální stabilitu (Monte Carlo efekt) přidáme drobný šum k exekuci
    for name, params in scenarios.items():
        results = []
        for _ in range(50): # 50 simulací s lehkou variací
            res = tester.run_scenario(
                cooldown=params[0], 
                threshold=params[1] + np.random.uniform(-0.005, 0.005), 
                penalty_mode=params[2]
            )
            results.append(res)
        
        final_stats[name] = results
        mean_pnl = np.mean(results)
        win_rate = sum(1 for r in results if r > 0) / len(results) * 100
        print(f"{name:25} | Mean PnL: ${mean_pnl:8.2f} | Win Rate: {win_rate:5.1f}%")

    # Vizuální výstup
    plt.figure(figsize=(12, 6))
    plt.style.use('ggplot')
    plt.boxplot(final_stats.values(), labels=final_stats.keys())
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Backtest na reálných datech: Dopad filtrů a 1% poplatků")
    plt.ylabel("Čistý zisk/ztráta ($)")
    plt.grid(True, axis='y', alpha=0.3)
    
    # Automatické uložení grafu
    plt.savefig("backtest_result.png")
    print("\nGraf byl uložen jako 'backtest_result.png'")
    plt.show()

if __name__ == "__main__":
    main()
