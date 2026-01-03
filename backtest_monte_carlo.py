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
        
        # Simulace 6 hodin (v 5s intervalech = 4320 kroků)
        prob = 0.50
        for t in range(4320):
            prob += np.random.normal(0, self.volatility * np.sqrt(interval/900))
            prob = np.clip(prob, 0.1, 0.9)
            
            # Logika signálu
            signal_strength = abs(prob - 0.50)
            can_trade = (t - entry_t) * interval >= cooldown if in_pos else True
            
            if not in_pos and signal_strength > (threshold - 0.50):
                in_pos, entry_t = True, t
                side = "UP" if prob > 0.50 else "DOWN"
                entry_p = (prob if side == "UP" else (1-prob)) * (1 + SLIPPAGE)
            
            elif in_pos and can_close := ((t - entry_t) * interval >= cooldown):
                # Zavřeme, pokud se trend otočil o víc než 2%
                if (side == "UP" and prob < 0.49) or (side == "DOWN" and prob > 0.51):
                    exit_p = (prob if side == "UP" else (1-prob)) * (1 - SLIPPAGE)
                    trade_pnl = (exit_p - entry_p) * (TRADE_SIZE / entry_p)
                    
                    # Aplikace asymetrické penalizace z run.py
                    if penalty_mode:
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
        win_rate = sum(1 for r in data if r > 0) / ITERATIONS * 100
        print(f"{name}: Mean=${np.mean(data):.2f}, Win={win_rate:.1f}%")

    plt.figure(figsize=(10, 6))
    plt.boxplot(results.values(), labels=results.keys())
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Monte Carlo: Survival with 1% Slippage")
    plt.ylabel("Profit/Loss ($)")
    plt.show()

if __name__ == "__main__": main()
