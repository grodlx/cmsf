import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import copy

# Simulace parametrů, které chceš otestovat
TEST_CONFIGS = {
    'iterations': 100,         # Počet simulací pro každý scénář
    'initial_balance': 1000,   # Počáteční kapitál v USD
    'trade_size': 500          # Velikost jedné pozice
}

class MonteCarloSimulator:
    def __init__(self, data_path=None):
        # Pokud nemáš CSV, simulujeme pohyb trhu (GBM - Geometric Brownian Motion)
        self.base_prob = 0.50
        self.volatility = 0.02 # 2% volatilita pro 15m okno
        
    def run_scenario(self, cooldown, threshold, tick_interval):
        """Simuluje jeden obchodní den s danými parametry."""
        balance = TEST_CONFIGS['initial_balance']
        pnl_history = []
        in_position = False
        entry_price = 0
        entry_time = 0
        side = None
        
        # Simulujeme 2880 kroků (cca 4 hodiny při 5s tiku)
        for t in range(2880):
            # 1. Simulace pohybu ceny s náhodným šumem (Monte Carlo prvek)
            current_prob = self.base_prob + np.random.normal(0, self.volatility)
            current_prob = np.clip(current_prob, 0.01, 0.99)
            
            # 2. Logika filtrů
            # A) Threshold: Ignorujeme šum kolem 0.50
            is_strong_signal = abs(current_prob - 0.50) > (threshold - 0.50)
            
            # B) Cooldown: Můžeme zavřít až po uplynutí času
            can_close = (t - entry_time) * tick_interval >= cooldown if in_position else True
            
            # 3. Exekuce (se zahrnutím tvého 1% slippage)
            if not in_position and is_strong_signal:
                in_position = True
                entry_time = t
                side = "UP" if current_prob > 0.50 else "DOWN"
                # Nákup s 1% slippage
                entry_price = current_prob * 1.01 if side == "UP" else (1 - current_prob) * 1.01
                
            elif in_position and can_close:
                # Rozhodnutí o uzavření (např. signál se obrátil)
                should_close = (side == "UP" and current_prob < 0.50) or (side == "DOWN" and current_prob > 0.50)
                
                if should_close:
                    exit_price_raw = current_prob if side == "UP" else (1 - current_prob)
                    # Prodej s 1% slippage
                    exit_price = exit_price_raw * 0.99
                    
                    shares = TEST_CONFIGS['trade_size'] / entry_price
                    trade_pnl = (exit_price - entry_price) * shares
                    balance += trade_pnl
                    pnl_history.append(trade_pnl)
                    in_position = False

        return balance - TEST_CONFIGS['initial_balance']

def analyze_strategies():
    sim = MonteCarloSimulator()
    scenarios = {
        "Baseline (No Filters)": {"cooldown": 0, "threshold": 0.50, "interval": 0.5},
        "Filtered (Cooldown 30s)": {"cooldown": 30, "threshold": 0.50, "interval": 5.0},
        "Aggressive (Threshold 0.55)": {"cooldown": 30, "threshold": 0.55, "interval": 5.0}
    }
    
    final_results = {}

    for name, params in scenarios.items():
        runs = [sim.run_scenario(params['cooldown'], params['threshold'], params['interval']) 
                for _ in range(TEST_CONFIGS['iterations'])]
        final_results[name] = runs
        print(f"Finishing {name}: Mean PnL = ${np.mean(runs):.2f}, Win Rate = {np.mean([1 for r in runs if r > 0]):.2%}")

    # Vizualizace
    plt.figure(figsize=(12, 6))
    plt.boxplot(final_results.values(), labels=final_results.keys())
    plt.title("Monte Carlo Backtest: Comparison of Filters (including 1% Slippage)")
    plt.ylabel("Net Profit ($)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    analyze_strategies()
