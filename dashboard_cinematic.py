#!/usr/bin/env python3
"""
Cinematic Dashboard v2.1 - With Live Unrealized PnL Tracking.
"""
import threading
import time
from datetime import datetime, timezone
from typing import Dict
from flask import Flask, render_template_string
from flask_socketio import SocketIO

# Global state
class DashboardState:
    def __init__(self):
        self.strategy_name = "ppo"
        self.total_pnl = 0.0
        self.total_unrealized_pnl = 0.0 # Nový sledovaný parametr
        self.trade_count = 0
        self.win_count = 0
        self.positions: Dict[str, dict] = {}
        self.markets: Dict[str, dict] = {}
        self.buffer_size = 0
        self.max_buffer = 2048
        self.updates = 0
        self.entropy = 0.0
        self.avg_reward = 0.0

dashboard_state = DashboardState()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cinematic'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Master Sniper Cinematic Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg: #050505;
            --surface: #0a0a0a;
            --border: #151515;
            --text: #e0e0e0;
            --dim: #444;
            --green: #00ff88;
            --red: #ff3355;
            --blue: #3388ff;
            --amber: #ffaa00;
        }

        body {
            font-family: 'JetBrains Mono', monospace;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            overflow: hidden;
        }

        .container {
            display: grid;
            grid-template-columns: 1fr 320px;
            grid-template-rows: 80px 1fr 200px;
            height: 100vh;
            gap: 1px;
            background: var(--border);
        }

        .header {
            grid-column: 1 / -1;
            background: var(--surface);
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 32px;
        }

        .logo h1 {
            font-size: 13px;
            font-weight: 500;
            color: var(--dim);
            letter-spacing: 3px;
            text-transform: uppercase;
        }

        .live-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.2);
        }

        .live-dot {
            width: 6px; height: 6px;
            background: var(--green);
            animation: pulse 1.5s infinite;
        }

        .header-stats { display: flex; gap: 48px; }
        .header-stat { text-align: right; }
        .header-stat-value { font-size: 32px; font-weight: 600; font-variant-numeric: tabular-nums; }
        .header-stat-value.positive { color: var(--green); }
        .header-stat-value.negative { color: var(--red); }
        .header-stat-label { font-size: 10px; color: var(--dim); text-transform: uppercase; }

        /* UPNL Specific styling */
        .upnl-display {
            font-size: 12px;
            font-weight: 600;
            margin-top: -4px;
        }
        .upnl-display.positive { color: var(--green); opacity: 0.8; }
        .upnl-display.negative { color: var(--red); opacity: 0.8; }

        .chart-area { background: var(--surface); padding: 24px; display: flex; flex-direction: column; }
        .chart-container { flex: 1; position: relative; min-height: 0; }
        #pnl-chart { width: 100%; height: 100%; }

        .sidebar { background: var(--surface); display: flex; flex-direction: column; overflow: hidden; }
        .sidebar-header { padding: 16px 20px; border-bottom: 1px solid var(--border); font-size: 11px; color: var(--dim); text-transform: uppercase; }
        .trades-list { flex: 1; overflow-y: auto; padding: 8px; }

        .trade-item { display: flex; align-items: center; padding: 12px; margin-bottom: 4px; background: var(--bg); gap: 12px; border-left: 2px solid var(--dim); }
        .trade-item.win { border-left-color: var(--green); }
        .trade-item.loss { border-left-color: var(--red); }
        
        .trade-side { font-size: 9px; font-weight: 600; padding: 4px 8px; text-transform: uppercase; }
        .trade-side.long { background: rgba(0,255,136,0.15); color: var(--green); }
        .trade-side.short { background: rgba(255,51,85,0.15); color: var(--red); }

        .markets-strip { grid-column: 1 / -1; background: var(--surface); display: flex; gap: 1px; overflow-x: auto; }
        .market-card { flex: 1; min-width: 200px; padding: 16px 20px; background: var(--bg); position: relative; }
        .market-card.has-position { background: #0e0e0e; border-top: 2px solid var(--blue); }

        .market-top { display: flex; justify-content: space-between; margin-bottom: 12px; }
        .market-timer { font-size: 18px; font-weight: 600; color: var(--text); }
        .market-mid { display: flex; align-items: baseline; gap: 12px; }
        .market-prob { font-size: 32px; font-weight: 700; }
        
        .market-position { display: flex; justify-content: space-between; padding: 8px 10px; font-size: 11px; margin-top: 8px; border-radius: 2px; }
        .market-position.long { background: rgba(0,255,136,0.1); color: var(--green); }
        .market-position.short { background: rgba(255,51,85,0.1); color: var(--red); }

        .time-progress { position: absolute; bottom: 0; left: 0; height: 2px; background: var(--blue); opacity: 0.3; }

        .stats-row { display: flex; gap: 1px; min-width: 400px; background: var(--border); }
        .stat-cell { flex: 1; padding: 16px; background: var(--bg); text-align: center; }
        .stat-value { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
        .stat-label { font-size: 9px; color: var(--dim); text-transform: uppercase; }

        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="logo">
                <h1>Master Sniper Cinematic</h1>
                <div class="live-indicator"><div class="live-dot"></div><span class="live-text">Live</span></div>
            </div>
            <div class="header-stats">
                <div class="header-stat">
                    <div class="header-stat-value" id="trades">0</div>
                    <div class="header-stat-label">Trades</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-value" id="winrate">0%</div>
                    <div class="header-stat-label">Win Rate</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-value" id="pnl">+$0.00</div>
                    <div id="upnl-total" class="upnl-display">uPnL: $0.00</div>
                    <div class="header-stat-label">Total Equity</div>
                </div>
            </div>
        </header>

        <div class="chart-area">
            <div class="chart-container"><canvas id="pnl-chart"></canvas></div>
        </div>

        <div class="sidebar">
            <div class="sidebar-header">Order History</div>
            <div class="trades-list" id="trades-list"></div>
        </div>

        <div class="markets-strip">
            <div id="markets-container" style="display:flex;gap:1px;flex:1;"></div>
            <div class="stats-row">
                <div class="stat-cell"><div class="stat-value" id="updates">0</div><div class="stat-label">Steps</div></div>
                <div class="stat-cell"><div class="stat-value" id="reward">0.00</div><div class="stat-label">Reward</div></div>
                <div class="stat-cell"><div class="stat-value" id="total-equity-stat">0.00</div><div class="stat-label">Net PnL</div></div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const socket = io();
        let pnlChart;
        let pnlHistory = [];
        let trades = [];

        function initChart() {
            const ctx = document.getElementById('pnl-chart').getContext('2d');
            pnlChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Net Equity',
                        data: [],
                        borderColor: '#00ff88',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        x: { display: false },
                        y: { position: 'right', grid: { color: '#151515' }, ticks: { color: '#444' } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        function formatPnl(v) {
            return (v >= 0 ? '+' : '-') + '$' + Math.abs(v).toFixed(2);
        }

        socket.on('state_update', (d) => {
            const markets = d.markets || {};
            const positions = d.positions || {};
            let currentUnrealized = 0;

            // 1. Calculate Unrealized PnL from active positions
            const container = document.getElementById('markets-container');
            container.innerHTML = Object.keys(markets).map(cid => {
                const m = markets[cid];
                const pos = positions[cid];
                const hasPos = pos && pos.size > 0;
                let posPnl = 0;

                if (hasPos) {
                    const shares = pos.size / pos.entry_price;
                    if (pos.side === 'UP') {
                        posPnl = (m.prob - pos.entry_price) * shares;
                    } else {
                        posPnl = ((1 - m.prob) - pos.entry_price) * shares;
                    }
                    currentUnrealized += posPnl;
                }

                return `
                    <div class="market-card ${hasPos ? 'has-position' : ''}">
                        <div class="market-top">
                            <span class="market-asset">${m.asset}</span>
                            <span class="market-timer">${Math.floor(m.time_left)}m</span>
                        </div>
                        <div class="market-mid">
                            <span class="market-prob">${(m.prob * 100).toFixed(1)}</span>
                        </div>
                        ${hasPos ? `
                            <div class="market-position ${pos.side.toLowerCase()}">
                                <span>${pos.side}</span>
                                <span>${formatPnl(posPnl)}</span>
                            </div>
                        ` : '<div class="no-position">NO POS</div>'}
                    </div>
                `;
            }).join('');

            // 2. Update Global Stats
            const realizedPnl = d.total_pnl || 0;
            const netEquity = realizedPnl + currentUnrealized;

            const pnlEl = document.getElementById('pnl');
            pnlEl.textContent = formatPnl(netEquity);
            pnlEl.className = 'header-stat-value ' + (netEquity >= 0 ? 'positive' : 'negative');

            const upnlEl = document.getElementById('upnl-total');
            upnlEl.textContent = `uPnL: ${formatPnl(currentUnrealized)}`;
            upnlEl.className = 'upnl-display ' + (currentUnrealized >= 0 ? 'positive' : 'negative');
            
            document.getElementById('total-equity-stat').textContent = netEquity.toFixed(2);

            // 3. Update Chart with Net Equity
            pnlHistory.push(netEquity);
            if (pnlHistory.length > 100) pnlHistory.shift();
            pnlChart.data.labels = pnlHistory.map((_, i) => i);
            pnlChart.data.datasets[0].data = pnlHistory;
            pnlChart.data.datasets[0].borderColor = netEquity >= 0 ? '#00ff88' : '#ff3355';
            pnlChart.update('none');

            document.getElementById('trades').textContent = d.trade_count;
            const wr = d.trade_count > 0 ? (d.win_count / d.trade_count * 100).toFixed(0) : 0;
            document.getElementById('winrate').textContent = wr + '%';
        });

        socket.on('trade', (t) => {
            const list = document.getElementById('trades-list');
            const item = document.createElement('div');
            item.className = `trade-item ${t.pnl >= 0 ? 'win' : 'loss'}`;
            item.innerHTML = `
                <span class="trade-side ${t.action.toLowerCase().includes('up') ? 'long' : 'short'}">${t.asset}</span>
                <div class="trade-details">
                    <div class="trade-asset">${t.action}</div>
                    <div class="trade-meta">${t.time}</div>
                </div>
                <span class="trade-pnl ${t.pnl >= 0 ? 'positive' : 'negative'}">${formatPnl(t.pnl || 0)}</span>
            `;
            list.prepend(item);
        });

        socket.on('connect', initChart);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def emit_state():
    socketio.emit('state_update', {
        'strategy_name': dashboard_state.strategy_name,
        'total_pnl': dashboard_state.total_pnl,
        'trade_count': dashboard_state.trade_count,
        'win_count': dashboard_state.win_count,
        'positions': dashboard_state.positions,
        'markets': dashboard_state.markets,
    })

def emit_rl_buffer(buffer_size: int, max_buffer: int = 2048, avg_reward: float = None):
    socketio.emit('rl_buffer', {'buffer_size': buffer_size, 'avg_reward': avg_reward})

def emit_trade(action: str, asset: str, size: float = 0, pnl: float = None):
    socketio.emit('trade', {
        'action': action, 'asset': asset, 'size': size, 'pnl': pnl,
        'time': datetime.now().strftime('%H:%M:%S'),
    })

def state_emitter():
    while True:
        time.sleep(0.4) # Rychlá aktualizace pro plynulé uPnL
        emit_state()

def update_dashboard_state(strategy_name=None, total_pnl=None, trade_count=None, win_count=None, positions=None, markets=None):
    if strategy_name is not None: dashboard_state.strategy_name = strategy_name
    if total_pnl is not None: dashboard_state.total_pnl = total_pnl
    if trade_count is not None: dashboard_state.trade_count = trade_count
    if win_count is not None: dashboard_state.win_count = win_count
    if positions is not None: dashboard_state.positions = positions
    if markets is not None: dashboard_state.markets = markets

def run_dashboard(host='0.0.0.0', port=5051):
    threading.Thread(target=state_emitter, daemon=True).start()
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    run_dashboard()
