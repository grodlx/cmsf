import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === KONFIGURACE ===
INITIAL_CAPITAL = 1000
TRADE_SIZE = 500 # Nastaveno podle vašeho posledního spuštění
SLIPPAGE = 0.01  # 1% (shodné s run.py)
FIXED_PENALTY = 0.15 

class LatestTradeBacktester:
    def __init__(self):
        self.filename = self._find_latest_trades()
        self.data = pd.read_csv(self.filename)
        print(f"Používám data ze souboru: {self.filename}")
        print(f"Načteno {len(self.data)} záznamů o obchodech.")

    def _find_latest_trades(self):
        # Najde nejnovější soubor začínající na 'trades' a končící '.csv'
        files = glob.glob("*trades*.csv")
        if not files:
            raise FileNotFoundError("Nebyl nalezen žádný soubor 'trades_*.csv'!")
        return max(files, key=os.path.getctime)

    def run_scenario(self, cooldown_ticks, threshold, penalty_mode=False):
        balance = INITIAL_CAPITAL
        in_pos = False
        entry_p, entry_t, side = 0, 0, None
        
        # Jako cenový feed použijeme prob_at_exit, což je čistá cena z trhu
        prices = self.data['prob_at_exit'].values
        
        for t, mid_price in enumerate(prices):
            signal_strength = abs(mid_price - 0.50)
            
            # LOGIKA VSTUPU
            if not in_pos:
                # Vstupujeme pouze pokud signál překoná threshold
                if signal_strength > (threshold - 0.50):
                    in_pos = True
                    entry_t = t
                    side = "UP" if mid_price > 0.50 else "DOWN"
                    # Nákupní cena s 1% slippage
                    entry_p = (mid_price if side == "UP" else (1 - mid_price)) * (1 + SLIPPAGE)
            
            # LOGIKA VÝSTUPU
            elif in_pos:
                # Cooldown (v tiku trade logu)
                if (t - entry_t) >= cooldown_ticks:
                    # Výstup při otočení trendu
                    should_close = (side == "UP" and mid_price < 0.49) or (side == "DOWN" and mid_price > 0.51)
                    
                    if should_close:
                        exit_p_raw = mid_price if side == "UP" else (1 - mid_price)
                        exit_p = exit_p_raw * (1 - SLIPPAGE)
                        
                        shares = TRADE_SIZE / entry_p
                        trade_pnl = (exit_p - entry_p) * shares
                        
                        if penalty_mode:
                            # Asymetrická penalizace
                            trade_pnl = (trade_pnl - FIXED_PENALTY) if trade_pnl > 0 else (trade_pnl * 1.5 - FIXED_PENALTY)
                        
                        balance += trade_pnl
                        in_pos = False
        
        return balance - INITIAL_CAPITAL

def main():
    try:
        tester = LatestTradeBacktester()
    except Exception as e:
        print(f"Chyba: {e}")
        return

    scenarios = {
        "Real Data: No Filters": (0, 0.50, False),
        "Real Data: Aggressive": (10, 0.55, False),
        "Real Data: Ultra (Penalized)": (20, 0.57, True)
    }
    
    final_stats = {}
    print("\n--- VÝSLEDKY SIMULACE ---")
    
    for name, params in scenarios.items():
        results = []
        # Uděláme 100 variací s mírným šumem v exekuci pro Monte Carlo efekt
        for _ in range(100):
            res = tester.run_scenario(
                cooldown_ticks=params[0], 
                threshold=params[1] + np.random.uniform(-0.005, 0.005), 
                penalty_mode=params[2]
            )
            results.append(res)
        
        final_stats[name] = results
        print(f"{name:25} | Průměrné PnL: ${np.mean(results):8.2f} | Úspěšnost: {sum(1 for r in results if r > 0)}%")

    plt.figure(figsize=(12, 6))
    plt.boxplot(final_stats.values(), labels=final_stats.keys())
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Backtest na datech: {tester.filename}")
    plt.ylabel("Zisk / Ztráta ($)")
    plt.show()

if __name__ == "__main__":
    main()
