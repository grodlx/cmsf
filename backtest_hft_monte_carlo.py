import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

# === KONFIGURACE SIMULACE ===
INITIAL_CAPITAL = 1000
TRADE_SIZE = 500
SLIPPAGE = 0.002       # Sníženo na 0.2% pro reálnější HFT odhad
FIXED_PENALTY = 0.10   # Fixní poplatek na obchod
ATR_WINDOW = 20        # Perioda pro výpočet volatility
SIMULATIONS_COUNT = 50 # Počet průchodů pro Monte Carlo statistiku

class MasterSniperMonteCarlo:
    def __init__(self, log_pattern="logs/market_ticks_hft_*.csv"):
        self.filename = self._find_latest_file(log_pattern)
        print(f"Načítám data pro simulaci: {self.filename}")
        self.df = pd.read_csv(self.filename)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['asset', 'timestamp'])

    def _find_latest_file(self, pattern):
        files = glob.glob(pattern)
        if not files: 
            # Fallback pokud složka neexistuje
            files = glob.glob("*.csv")
        return max(files, key=os.path.getmtime) if files else None

    def run_simulation(self, threshold, tick_step=10, min_hold_sec=30, 
                       tp_mult=None, sl_mult=None, use_penalty=True):
        """
        Spustí backtest na historických datech.
        Pokud je tp_mult/sl_mult definován, použije dynamický výstup.
        """
        total_balance = INITIAL_CAPITAL
        
        for asset, group in self.df.groupby('asset'):
            group = group.copy()
            # Dynamický odhad volatility (ATR-like z 'prob')
            group['atr'] = group['prob'].diff().abs().rolling(window=ATR_WINDOW).mean()
            
            sampled_group = group.iloc[::tick_step]
            prices = sampled_group['prob'].values
            atrs = sampled_group['atr'].values
            times = sampled_group['timestamp'].values
            
            pos = None
            
            for t in range(len(prices)):
                mid_price = prices[t]
                current_atr = atrs[t] if not np.isnan(atrs[t]) else 0.0005
                current_time = times[t]
                
                if pos is None:
                    # VSTUPNÍ LOGIKA (Threshold)
                    dist = abs(mid_price - 0.50)
                    if dist > (threshold - 0.50):
                        side = "UP" if mid_price > 0.50 else "DOWN"
                        # Cena po slippage
                        entry_p = (mid_price if side == "UP" else (1 - mid_price)) * (1 + SLIPPAGE)
                        
                        pos = {
                            'entry_p': entry_p, 
                            'entry_t': current_time, 
                            'side': side,
                            'tp_level': entry_p + (current_atr * tp_mult) if tp_mult else None,
                            'sl_level': entry_p - (current_atr * sl_mult) if sl_mult else None
                        }
                else:
                    # VÝSTUPNÍ LOGIKA
                    duration_sec = (current_time - pos['entry_t']) / np.timedelta64(1, 's')
                    curr_p_raw = mid_price if pos['side'] == "UP" else (1 - mid_price)
                    
                    is_tp = pos['tp_level'] and curr_p_raw >= pos['tp_level']
                    is_sl = pos['sl_level'] and curr_p_raw <= pos['sl_level']
                    is_reversal = duration_sec >= min_hold_sec and (
                        (pos['side'] == "UP" and mid_price < 0.50) or 
                        (pos['side'] == "DOWN" and mid_price > 0.50)
                    )

                    if is_tp or is_sl or is_reversal:
                        exit_price = curr_p_raw * (1 - SLIPPAGE)
                        shares = TRADE_SIZE / pos['entry_p']
                        pnl = (exit_price - pos['entry_p']) * shares
                        
                        if use_penalty:
                            # Asymetrická penalizace (tvůj model)
                            pnl = (pnl - FIXED_PENALTY) if pnl > 0 else (pnl * 1.3 - FIXED_PENALTY)
                        
                        total_balance += pnl
                        pos = None
                            
        return total_balance - INITIAL_CAPITAL

def main():
    tester = MasterSniperMonteCarlo()
    
    # Scénáře pro porovnání
    # Formát: (Threshold, TickStep, MinHold, TP_mult, SL_mult, Penalty)
    scenarios = {
        "Původní (No TP)": (0.62, 10, 120, None, None, True),
        "Sniper + ATR TP (1.5x)": (0.62, 10, 30, 1.5, 1.0, True),
        "Agresivní TP (0.8x)": (0.60, 5, 10, 0.8, 0.8, True)
    }

    final_results = {}
    print("\n" + "="*70)
    print(f"{'STRATEGIE':25} | {'MEAN PnL':10} | {'WIN RATE':10} | {'MAX LOSS':10}")
    print("="*70)

    for name, params in scenarios.items():
        # Monte Carlo: Spustíme simulaci vícekrát pro statistickou stabilitu
        data = [tester.run_simulation(*params) for _ in range(SIMULATIONS_COUNT)]
        final_results[name] = data
        
        mean_pnl = np.mean(data)
        win_rate = sum(1 for r in data if r > 0) / len(data) * 100
        max_loss = np.min(data)
        
        print(f"{name:25} | ${mean_pnl:8.2f} | {win_rate:8.1f}% | ${max_loss:8.2f}")

    # Vizualizace
    plt.figure(figsize=(12, 6))
    plt.boxplot(final_results.values(), labels=final_results.keys())
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.title(f"Monte Carlo Verifikace: Master Sniper 5s (N={SIMULATIONS_COUNT})")
    plt.ylabel("Zisk / Ztráta (USD)")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
