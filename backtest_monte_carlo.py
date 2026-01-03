import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# CONFIG
ITERATIONS = 100
INITIAL_CAPITAL = 1000
TRADE_SIZE = 500
SLIPPAGE = 0.01  # 1%

class MonteCarloSimulator:
    def __init__(self):
        self.volatility = 0.015

    def run_scenario(self, cooldown, threshold, interval, penalty_mode=False):
        balance = INITIAL_CAPITAL
        in_pos = False
        entry_p, entry_t, side = 0, 0, None
        pnl_history = []
        
        # Simulace 6 hodin (v intervalech dle parametru)
        prob = 0.50
        # Počet kroků (6h = 21600s)
        steps = int(21600 / interval)
        
        for t in range(steps):
            # Simulace pohybu ceny
            prob += np.random.normal(0, self.volatility * np.sqrt(interval/900))
            prob = np.clip(prob, 0.1, 0.9)
            
            signal_strength = abs(prob - 0.50)
            
            # LOGIKA VSTUPU
            if not in_pos:
                if signal_strength > (threshold - 0.50):
                    in_pos = True
                    entry_t = t
                    side = "UP" if prob > 0.50 else "DOWN"
                    # Nákup s 1% slippage (vstupní cena je horší o 1%)
                    entry_p = (prob if side == "UP" else (1-prob)) * (1 + SLIPPAGE)
            
            # LOGIKA VÝSTUPU
            elif in_pos:
                # Kontrola cooldownu
                if (t - entry_t) * interval >= cooldown:
                    # Zavřeme, pokud se trend otočil nebo vypršel signál
                    should_close = (side == "UP" and prob < 0.49) or (side == "DOWN" and prob > 0.51)
                    
                    if should_close:
                        # Prodej s 1% slippage (výstupní cena je horší o 1%)
                        exit_p_raw = prob if side == "UP" else (1-prob)
                        exit_p = exit_p_raw * (1 - SLIPPAGE)
                        
                        shares = TRADE_SIZE / entry_p
                        trade_pnl = (exit_p - entry_p) * shares
                        
                        # Aplikace asymetrické penalizace z run.py
                        if penalty_mode:
                            # Pokud je zisk kladný, odečteme fixní náklad. 
                            # Pokud záporný, násobíme ho 1.5x (trest za chybu).
                            trade_pnl = (trade_pnl - 0.15) if trade_pnl > 0 else (trade_pnl * 1.5 - 0.15)
                        
                        balance += trade_pnl
                        pnl_history.append(trade_pnl)
                        in_pos = False
        
        return balance - INITIAL_CAPITAL

def main():
    sim = MonteCarloSimulator()
    scenarios = {
        "Baseline (0.5s)": (0, 0.50, 0.5, False),
        "Aggressive (30s)": (30, 0.55, 5.0, False),
        "Ultra-Aggressive (60s+Pen)": (60, 0.56, 5.0, True)
    }
    
    results = {}
    for name, params in scenarios.items():
        data = [sim.run_scenario(*params) for _ in range(ITERATIONS)]
        results[name] = data
        wins = sum(1 for r in data if r > 0)
        win_rate = (wins / ITERATIONS) * 100
        print(f"{name}: Mean PnL = ${np.mean(data):.2f}, Win Rate = {win_rate:.1f}%")

    # Grafické zobrazení
    plt.figure(figsize=(12, 7))
    plt.boxplot(results.values(), labels=results.keys())
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Monte Carlo: Survival Strategy Comparison (with 1% Slippage)")
    plt.ylabel("Net Profit/Loss ($)")
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    main()
