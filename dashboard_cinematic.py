#!/usr/bin/env python3
import threading
import time
import os
from datetime import datetime
from typing import Dict
from flask import Flask, render_template_string
from flask_socketio import SocketIO

# --- GLOBÁLNÍ STAV ---
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
socketio = SocketIO(app, cors_allowed_origins="*")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Master Sniper Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body { background: #050505; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; margin: 0; overflow: hidden; }
        .grid { display: grid; grid-template-columns: 1fr 350px; grid-template-rows: 100px 1fr 220px; height: 100vh; }
        
        .header { grid-column: 1/-1; background: #0a0a0a; border-bottom: 1px solid #151515; display: flex; justify-content: space-between; align-items: center; padding: 0 40px; }
        .stats { display: flex; gap: 40px; }
        .stat-box { text-align: right; }
        .val { font-size: 28px; font-weight: bold; }
        .lab { font-size: 10px; color: #444; text-transform: uppercase; }
        
        .main { background: #050505; padding: 20px; position: relative; }
        .sidebar { background: #0a0a0a; border-left: 1px solid #151515; overflow-y: auto; padding: 20px; }
        .footer { grid-column: 1/-1; background: #0a0a0a; border-top: 1px solid #151515; display: flex; gap: 10px; padding: 10px; overflow-x: auto; }
        
        .card { min-width: 220px; background: #050505; border: 1px solid #151515; padding: 15px; border-radius: 4px; }
        .card.active { border-color: #3388ff; }
        
        .trade-log { font-size: 12px; padding: 10px; background: #050505; margin-bottom: 5px; border-radius: 4px; border-left: 3px solid #3388ff; }
        .pos { color: #00ff88; }
        .neg { color: #ff3355; }
    </style>
</head>
<body>
    <div class="grid">
        <div class="header">
            <div><b style="letter-spacing:2px">MASTER SNIPER</b><br><small id="strat-name">V3.0</small></div>
            <div class="stats">
                <div class="stat-stat"><div class="val" id="count">0</div><div class="lab">Trades</div></div>
                <div class="stat-stat"><div class="val" id="winrate">0%</div><div class="lab">Winrate</div></div>
                <div class="stat-stat"><div class="val" id="net-pnl">$0.00</div><div class="lab">Net Equity (uPnL)</div></div>
            </div>
        </div>
        <div class="main"><canvas id="chart"></canvas></div>
        <div class="sidebar" id="log"></div>
        <div class="footer" id="markets"></div>
    </div>

    <script>
        const socket = io();
        let chart;
        let history = [];

        const ctx = document.getElementById('chart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Equity', data: [], borderColor: '#00ff88', borderWidth: 2, pointRadius: 0, fill: true, backgroundColor: 'rgba(0,255,136,0.05)', tension: 0.4 }] },
            options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { x: { display: false }, y: { position: 'right', grid: { color: '#111' } } }, plugins: { legend: { display: false } } }
        });

        socket.on('state_update', (data) => {
            const markets = data.markets || {};
            const positions = data.positions || {};
            let upnl = 0;

            // Update Markets
            const mDiv = document.getElementById('markets');
            mDiv.innerHTML = Object.keys(markets).map(id => {
                const m = markets[id];
                const p = positions[id];
                const hasP = p && p.size > 0;
                if(hasP) {
                    const shares = p.size / p.entry_price;
                    const curP = (p.side === 'UP') ? m.prob : (1 - m.prob);
                    const curPnl = (curP - p.entry_price) * shares;
                    upnl += curPnl;
                }
                return `<div class="card ${hasP?'active':''}">
                    <div class="lab">${m.asset}</div>
                    <div class="val" style="font-size:20px">${(m.prob*100).toFixed(1)}%</div>
                    ${hasP?`<div class="${upnl>=0?'pos':'neg'}" style="font-size:12px"><b>${p.side}</b> $${p.size}</div>`:''}
                </div>`;
            }).join('');

            const net = (data.total_pnl || 0) + upnl;
            document.getElementById('net-pnl').textContent = '$' + net.toFixed(2);
            document.getElementById('net-pnl').className = 'val ' + (net >= 0 ? 'pos' : 'neg');
            document.getElementById('count').textContent = data.trade_count || 0;
            document.getElementById('winrate').textContent = (data.trade_count > 0 ? (data.win_count/data.trade_count*100).toFixed(0) : 0) + '%';

            history.push(net);
            if(history.length > 100) history.shift();
            chart.data.labels = history.map((_,i)=>i);
            chart.data.datasets[0].data = history;
            chart.update('none');
        });

        socket.on('trade', (t) => {
            const log = document.getElementById('log');
            const row = document.createElement('div');
            row.className = 'trade-log';
            row.style.borderLeftColor = t.pnl >= 0 ? '#00ff88' : '#ff3355';
            row.innerHTML = `<b>${t.asset}</b> ${t.action}<br><span class="${t.pnl>=0?'pos':'neg'}">$${(t.pnl||0).toFixed(2)}</span> <small style="float:right">${t.time}</small>`;
            log.prepend(row);
        });
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
    def emitter():
        while True:
            socketio.emit('state_update', {
                'total_pnl': dashboard_state.total_pnl,
                'trade_count': dashboard_state.trade_count,
                'win_count': dashboard_state.win_count,
                'positions': dashboard_state.positions,
                'markets': dashboard_state.markets
            })
            time.sleep(0.5)
    
    threading.Thread(target=emitter, daemon=True).start()
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    run_dashboard()
