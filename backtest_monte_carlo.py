import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# CONFIG
INITIAL_CAPITAL = 1000
TRADE_SIZE = 500
SLIPPAGE = 0.01  # 1%

class CSVBacktester:
    def __init__(self, log_pattern="logs/*.csv"):
        self.data = self._load_data(log_pattern)
        
    def _load_data(self, pattern):
        # Načte všechny CSV logy a spojí je
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError("Ve složce 'logs/' nebyly nalezeny žádné CSV soubory!")
        
        all_df = []
        for f in files:
            df = pd.read_csv(f)
            all_df.append(df)
        
        combined = pd.concat(all_df).sort_values(by=df.columns[0]) # Předpokládáme čas v 1. sloupci
        print(f"Načteno {len(combined)} záznamů z realných dat.")
        return combined

    def run_scenario(self, cooldown, threshold, penalty_mode=False):
        balance = INITIAL_CAPITAL
        in_pos = False
        entry_p, entry_t, side = 0, 0, None
        
        # Iterujeme skrze reálná data z logu
        # Použijeme prob_at_exit jako průběžnou cenu trhu
        prices = self.data['prob_at_exit'].values
        
        for t, prob in enumerate(prices):
            signal_strength = abs(prob - 0.50)
            
            # LOGIKA VSTUPU
            if not in_pos:
                if signal_strength > (threshold - 0.50):
                    in_pos = True
                    entry_t = t
                    side = "UP" if prob > 0.50 else "DOWN"
                    # Realita: Nákup s 1% slippage
                    entry_p = (prob if side == "UP" else (1-prob)) * (1 + SLIPPAGE)
            
            # LOGIKA VÝSTUPU
            elif in_pos:
                # Simulujeme "tiky" jako indexy v CSV (1 řádek = 1 tik)
                # Pokud logy nejsou vteřinové, cooldown zde funguje jako "počet řádků"
                if (t - entry_t) >= cooldown:
                    # Výstupní podmínka: signál se otočil
                    should_close = (side == "UP" and prob < 0.49) or (side == "DOWN" and prob > 0.51)
                    
                    if should_close:
                        exit_p_raw = prob if side == "UP" else (1-prob)
                        exit_p = exit_p_raw * (1 - SLIPPAGE)
                        
                        shares = TRADE_SIZE / entry_p
                        trade_pnl = (exit_p - entry_p) * shares
                        
                        if penalty_mode:
                            trade_pnl = (trade_pnl - 0.15) if trade_pnl > 0 else (trade_pnl * 1.5 - 0.15)
                        
                        balance += trade_pnl
                        in_pos = False
        
        return balance - INITIAL_CAPITAL

def main():
    try:
        tester = CSVBacktester()
    except Exception as e:
        print(f"Chyba: {e}")
        print("Ujistěte se, že jste nejdříve spustili 'run.py', aby se vygenerovaly logy v 'logs/'.")
        return

    scenarios = {
        "Real Data: No Filters": (0, 0.50, False),
        "Real Data: Cooldown 30 ticks": (30, 0.50, False),
        "Real Data: Ultra-Aggressive": (60, 0.56, True)
    }
    
    results = {}
    for name, params in scenarios.items():
        # U reálných dat dáváme jen 1 iteraci (nebo můžeme přidat náhodný šum pro Monte Carlo efekt)
        # Zde přidáme 50 iterací s drobným šumem (+- 0.5%), abychom viděli stabilitu na reálných datech
        data = []
        for _ in range(50):
            # Přidáme drobný šum k thresholdům, abychom simulovali nejistotu v exekuci
            res = tester.run_scenario(params[0], params[1] + np.random.normal(0, 0.005), params[2])
            data.append(res)
            
        results[name] = data
        win_rate = sum(1 for r in data if r > 0) / len(data) * 100
        print(f"{name}: Mean PnL = ${np.mean(data):.2f}, Win Rate = {win_rate:.1f}%")

    plt.figure(figsize=(12, 7))
    plt.boxplot(results.values(), labels=results.keys())
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Backtest na reálných datech z CSV logů (včetně 1% Slippage)")
    plt.ylabel("Zisk / Ztráta ($)")
    plt.show()

if __name__ == "__main__":
    main()
