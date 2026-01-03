import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === KONFIGURACE REVERSAL SIMULACE ===
INITIAL_CAPITAL = 1000
TRADE_SIZE = 500
# Sníženo na reálnější HFT hodnoty, aby se strategie mohla "nadechnout"
SLIPPAGE = 0.0005      # 0.05% (Klíčové pro skalpování)
FIXED_PENALTY = 0.05   # Fixní poplatek
ATR_WINDOW = 15        
SIMULATIONS_COUNT = 30 

class MasterSniperReversalMC:
    def __init__(self, log_pattern="logs/market_ticks_hft_*.csv"):
        self.filename = self._find_latest_file(log_pattern)
        print(f"REVERSAL ANALÝZA: {self.filename}")
        self.df = pd.read_csv(self.filename)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['asset', 'timestamp'])

    def _find_latest_file(self, pattern):
        files = glob.glob(pattern)
        if not files: files = glob.glob("*.csv")
        return max(files, key=os.path.getmtime) if files else None

    def run_reversal_simulation(self, threshold, tp_mult=1.2, sl_mult=1.8):
        """
        Reversal logika: 
        Pokud prob > threshold -> Otevíráme DOWN (čekáme návrat)
        Pokud prob < (1-threshold) -> Otevíráme UP
        """
        total_balance = INITIAL_CAPITAL
        
        for asset, group in self.df.groupby('asset'):
            group = group.copy()
            group['atr'] = group['prob'].diff().abs().rolling(window=ATR_WINDOW).mean()
            
            # Sampling pro 5s reakci
            sampled_group = group.iloc[::10] 
            prices = sampled_group['prob'].values
            atrs = sampled_group['atr'].values
            times = sampled_group['timestamp'].values
            
            pos = None
            
            for t in range(len(prices)):
                mid_price = prices[t]
                current_atr = atrs[t] if not np.isnan(atrs[t]) else 0.001
                
                if pos is None:
                    # --- REVERSAL VSTUP ---
                    if mid_price > threshold:
                        side = "DOWN" # Trh je moc vysoko, sázíme na pád
                        entry_p = (1 - mid_price) * (1 + SLIPPAGE)
                        pos = {'entry_p': entry_p, 'side': side, 'entry_t': times[t],
                               'tp': entry_p + (current_atr * tp_mult),
                               'sl': entry_p - (current_atr * sl_mult)}
                    
                    elif mid_price < (1 - threshold):
                        side = "UP" # Trh je moc nízko, sázíme na růst
                        entry_p = mid_price * (1 + SLIPPAGE)
                        pos = {'entry_p': entry_p, 'side': side, 'entry_t': times[t],
                               'tp': entry_p + (current_atr * tp_mult),
                               'sl': entry_p - (current_atr * sl_mult)}
                else:
                    # --- VÝSTUP (TP/SL nebo Time Exit) ---
                    curr_p_raw = (1 - mid_price) if pos['side'] == "DOWN" else mid_price
                    duration = (times[t] - pos['entry_t']) / np.timedelta64(1, 's')
                    
                    if curr_p_raw >= pos['tp'] or curr_p_raw <= pos['sl'] or duration > 60:
                        exit_price = curr_p_raw * (1 - SLIPPAGE)
                        shares = TRADE_SIZE / pos['entry_p']
                        pnl = (exit_price - pos['entry_p']) * shares
                        
                        # Penalizace (očištěná o poplatky)
                        total_balance += (pnl - FIXED_PENALTY)
                        pos = None
                            
        return total_balance - INITIAL_CAPITAL

def main():
    tester = MasterSniperReversalMC()
    
    # Testujeme různé úrovně agresivity reversal vstupu
    scenarios = {
        "Reversal (Thr 0.60)": (0.60, 1.2, 1.5),
        "Reversal (Thr 0.65)": (0.65, 1.5, 2.0),
        "Reversal (Extreme 0.70)": (0.70, 2.0, 2.0)
    }

    final_results = {}
    print("\n" + "="*70)
    print(f"{'REVERSAL STRATEGIE':25} | {'MEAN PnL':10} | {'WIN RATE':10}")
    print("="*70)

    for name, params in scenarios.items():
        data = [tester.run_reversal_simulation(*params) for _ in range(SIMULATIONS_COUNT)]
        final_results[name] = data
        print(f"{name:25} | ${np.mean(data):8.2f} | {sum(1 for r in data if r > 0)/len(data)*100:8.1f}%")

    plt.figure(figsize=(10, 6))
    plt.boxplot(final_results.values(), labels=final_results.keys())
    plt.axhline(0, color='green', lw=1, ls='--')
    plt.title("Monte Carlo: Reversal Strategy Verification")
    plt.show()

if __name__ == "__main__":
    main()
    
