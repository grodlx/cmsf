#!/usr/bin/env python3
"""
Cinematic Dashboard v2.2 - Full Fixed Version
With Real-time Unrealized PnL and Equity Curve.
"""
import threading
import time
from datetime import datetime, timezone
from typing import Dict
from flask import Flask, render_template_string
from flask_socketio import SocketIO

# Global state management
class DashboardState:
    def __init__(self):
        self.strategy_name = "Master Sniper"
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.positions: Dict[str, dict] = {}
        self.markets: Dict[str, dict] = {}

dashboard_state = DashboardState()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cinematic_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Master Sniper Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg: #050505; --surface: #0a0a0a; --border: #151515;
            --text: #e0e0e0; --dim: #444; --green: #00ff88;
            --red: #ff3355; --blue: #3388ff;
        }
        body {
            font-family: 'JetBrains Mono', monospace;
            background: var(--bg); color: var(--text);
            height: 100vh; overflow: hidden;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 350px;
            grid-template-rows: 100px 1fr 220px;
            height: 100vh; gap: 1px; background: var(--border);
        }
        .header {
            grid-column: 1 / -1; background: var(--surface);
            display: flex; justify-content: space-between; align-items: center; padding: 0 40px;
        }
        .header-stats { display: flex; gap: 60px; }
        .header-stat { text-align: right; }
        .stat-value { font-size: 32px; font-weight: 600; font-variant-numeric: tabular-nums; }
        .stat-label { font-size: 10px; color: var(--dim); text-transform: uppercase; letter-spacing: 1px; }
        .upnl-sub { font-size: 14px; font-weight: 600; margin-top: -5px; }
        
        .chart-area { background: var(--surface); padding: 30px; display: flex; flex-direction: column; }
        .chart-container { flex: 1; position: relative; }
        
        .sidebar { background: var(--surface); display: flex; flex-direction: column; overflow: hidden; }
        .sidebar-header { padding: 20px; border-bottom: 1px solid var(--border); font-size: 12px; color: var(--dim); text-transform: uppercase; }
        .trades-list { flex: 1; overflow-y: auto; padding: 10px; }
        .trade-item { 
            display: flex; justify-content: space-between; align-items: center; 
            padding: 15px; background: var(--bg); margin-bottom: 5px; border-radius: 4px;
        }
        
        .markets-strip { grid-column: 1 / -1; background: var(--surface); display: flex; overflow-x: auto; padding: 10px; gap: 10px; }
        .market-card { 
            min-width: 240px; padding: 20px; background: var(--bg); border-radius: 4px; border: 1px solid var(--border);
            position: relative;
        }
        .market-card.active { border-color: var(--blue); background: #0c0c0c; }
        .market-asset { font-size: 16px; font-weight: 600; margin-bottom: 10px; display: block; }
        .market-prob { font-size: 38px; font-weight: 700; color: var(--text); }
        .market-pos { 
            margin-top: 15px; padding: 8px; border-radius: 3px; font-size: 12px; 
            display: flex; justify-content: space-between; font-weight: 600;
        }
        .pos-up { background: rgba(0, 255, 136, 0.1); color: var(--green); }
        .pos-down { background: rgba(255, 51, 85, 0.1); color: var(--red); }
        
        .positive { color: var(--green); }
        .negative { color: var(--red); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h2 style="font-size: 14px; color: var(--dim); letter-spacing: 2px;">MASTER SNIPER ENGINE</h2>
                <div style="display:flex; align-items:center; gap:10px; margin-top:5px;">
                    <div style="width:8px; height:8px; background:var(--green); border-radius:50%;"></div>
                    <span style="font-size:12px; color:var(--green); text-transform:uppercase;">Live Feed</span>
                </div>
            </div>
            <div class="header-stats">
                <div class="header-stat">
                    <div class="stat-value" id="stat-trades">0</div>
                    <div class="stat-label">Total Trades</div>
                </div>
                <div class="header-stat">
                    <div class="stat-value" id="stat-winrate">0%</div>
                    <div class="stat-label">Win Rate</div>
                </div>
                <div class="header-stat">
                    <div class="stat-value positive" id="stat-pnl">$0.00</div>
                    <div class="upnl-sub" id="stat-upnl">uPnL: $0.00</div>
                    <div class="stat-label">Net Equity</div>
                </div>
            </div>
        </div>

        <div class="chart-area">
            <div class="chart-container"><canvas id="main-chart"></canvas></div>
        </div>

        <div class="sidebar">
            <div class="sidebar-header">Recent Executions</div>
            <div id="trades-list" class="trades-list"></div>
        </div>

        <div class="markets-strip" id="markets-container"></div>
    </div>

    <script>
        const socket = io();
        let chart;
        let pnlHistory = [];

        function initChart() {
            const ctx = document.getElementById('main-chart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Equity', data: [], borderColor: '#00ff88',
                        borderWidth: 2, tension: 0.4, pointRadius: 0, fill: true,
                        backgroundColor: 'rgba(0, 255, 136, 0.05)'
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false, animation: false,
                    scales: {
                        x: { display: false },
                        y: { position: 'right', grid: { color: '#111' }, ticks: { color: '#444' } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        socket.on('state_update', (data) => {
            const markets = data.markets || {};
            const positions = data.positions || {};
            let totalUnrealized = 0;

            // Update Markets and Calculate uPnL
            const container = document.getElementById('markets-container');
            container.innerHTML = Object.keys(markets).map(cid => {
                const m = markets[cid];
                const pos = positions[cid];
                const hasPos = pos && pos.size > 0;
                let upnl = 0;

                if (hasPos) {
                    const shares = pos.size / pos.entry_price;
                    upnl = (pos.side === 'UP') ? 
                        (m.prob - pos.entry_price) * shares : 
                        ((1 - m.prob) - pos.entry_price) * shares;
                    totalUnrealized += upnl;
                }

                return `
                    <div class="market-card ${hasPos ? 'active' : ''}">
                        <span class="market-asset">${m.asset}</span>
                        <div class="market-prob">${(m.prob * 100).toFixed(1)}</div>
                        ${hasPos ? `
                            <div class="market-pos ${pos.side === 'UP' ? 'pos-up' : 'pos-down'}">
                                <span>${pos.side}</span>
                                <span>${upnl >= 0 ? '+' : ''}${upnl.toFixed(2)}</span>
                            </div>
                        ` : '<div style="color:#222; font-size:12px; margin-top:15px;">NO POSITION</div>'}
                    </div>
                `;
            }).join('');

            // Update Header
            const netEquity = data.total_pnl + totalUnrealized;
            const pnlEl = document.getElementById('stat-pnl');
            pnlEl.textContent = (netEquity >= 0 ? '+' : '-') + '$' + Math.abs(netEquity).toFixed(2);
            pnlEl.className = 'stat-value ' + (netEquity >= 0 ? 'positive' : 'negative');

            const upnlEl = document.getElementById('stat-upnl');
            upnlEl.textContent = 'uPnL: ' + (totalUnrealized >= 0 ? '+' : '-') + '$' + Math.abs(totalUnrealized).toFixed(2);
            upnlEl.className = 'upnl-sub ' + (totalUnrealized >= 0 ? 'positive' : 'negative');

            document.getElementById('stat-trades').textContent = data.trade_count;
            const wr = data.trade_count > 0 ? (data.win_count / data.trade_count * 100).toFixed(0) : 0;
            document.getElementById('stat-winrate').textContent = wr + '%';

            // Update Chart
            pnlHistory.push(netEquity);
            if (pnlHistory.length > 100) pnlHistory.shift();
            chart.data.labels = pnlHistory.map((_, i) => i);
            chart.data.datasets[0].data = pnlHistory;
            chart.update('none');
        });

        socket.on('trade', (t) => {
            const list = document.getElementById('trades-list');
            const row = document.createElement('div');
            row.className = 'trade-item';
            row.innerHTML = `
                <div>
                    <div style="font-size:13px; font-weight:600;">${t.asset}</div>
                    <div style="font-size:10px; color:#444;">${t.time}</div>
                </div>
                <div class="${t.pnl >= 0 ? 'positive' : 'negative'}" style="font-weight:600;">
                    ${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}
                </div>
            `;
            list.prepend(row);
        });

        window.onload = initChart;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def update_dashboard_state(strategy_name=None, total_pnl=None, trade_count=None, win_count=None, positions=None, markets=None):
    if strategy_name is not None: dashboard_state.strategy_name = strategy_name
    if total_pnl is not None: dashboard_state.total_pnl = total_pnl
    if trade_count is not None: dashboard_state.trade_count = trade_count
    if win_count is not None: dashboard_state.win_count = win_count
    if positions is not None: dashboard_state.positions = positions
    if markets is not None: dashboard_state.markets = markets

def emit_trade(action: str, asset: str, size: float = 0, pnl: float = None):
    socketio.emit('trade', {
        'action': action, 'asset': asset, 'size': size, 'pnl': pnl or 0,
        'time': datetime.now().strftime('%H:%M:%S'),
    })

def run_dashboard(host='0.0.0.0', port=5051):
    def state_emitter():
        while True:
            socketio.emit('state_update', {
                'total_pnl': dashboard_state.total_pnl,
                'trade_count': dashboard_state.trade_count,
                'win_count': dashboard_state.win_count,
                'positions': dashboard_state.positions,
                'markets': dashboard_state.markets
            })
            time.sleep(0.4)
    
    threading.Thread(target=state_emitter, daemon=True).start()
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    run_dashboard()
