#!/usr/bin/env python3
"""
generate_dashboard.py — Build dashboard.html from paper trader state.

Reads paper_state.json + paper_trades.jsonl + paper_signals.jsonl and outputs
an HTML dashboard with PnL stats, equity curve, trade list, open positions.

Designed to be regenerated on every workflow cycle and served via GitHub Pages.
"""
import json
import datetime
from pathlib import Path
from html import escape

HERE = Path(__file__).parent
STATE_FILE = HERE / "paper_state.json"
TRADES_FILE = HERE / "paper_trades.jsonl"
SIGNALS_FILE = HERE / "paper_signals.jsonl"
OUT = HERE / "dashboard.html"

def load_jsonl(path):
    out = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {}

state = load_state()
trades = load_jsonl(TRADES_FILE)
signals = load_jsonl(SIGNALS_FILE)

now = datetime.datetime.now(datetime.timezone.utc)
now_str = now.strftime("%Y-%m-%d %H:%M UTC")

# ── Stats ────────────────────────────────────────────────────────
closed_count = len(trades)
open_positions = state.get("positions", {})
open_count = len(open_positions)
cycles_run = state.get("cycle", 0)
started_at = state.get("started_at", "—")

total_pnl = sum(t.get("net_pnl_usd", 0) for t in trades)
total_fees = sum(t.get("total_fees_usd", 0) for t in trades)
wins = sum(1 for t in trades if t.get("net_pnl_usd", 0) > 0)
win_rate = (wins / closed_count * 100) if closed_count else 0
avg_trade = (total_pnl / closed_count) if closed_count else 0
best = max((t.get("net_pnl_usd", 0) for t in trades), default=0)
worst = min((t.get("net_pnl_usd", 0) for t in trades), default=0)

# Backtest expectations (from extended_v3 analysis)
EXP_WIN_RATE = 54.3
EXP_AVG_TRADE = 7.66

# ── Equity curve data ────────────────────────────────────────────
sorted_trades = sorted(trades, key=lambda t: t.get("exit_ts", 0))
equity_x = []
equity_y = []
running = 0
for t in sorted_trades:
    running += t.get("net_pnl_usd", 0)
    ts = t.get("exit_ts", 0)
    equity_x.append(datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M"))
    equity_y.append(round(running, 2))

# Per-trade PnL histogram bins
pnls = [t.get("net_pnl_usd", 0) for t in trades]

# ── HTML ─────────────────────────────────────────────────────────
def color_pnl(v):
    if v > 0: return "#2ea043"
    if v < 0: return "#cf222e"
    return "#666"

open_rows = ""
for mid, pos in open_positions.items():
    held_h = (now.timestamp() - pos.get("entry_ts", 0)) / 3600
    open_rows += f"""
        <tr>
            <td>{escape(str(pos.get('question', ''))[:80])}</td>
            <td>{pos.get('entry_z', 0):+.2f}</td>
            <td>{'short' if pos.get('direction') == -1 else 'long'}</td>
            <td>${pos.get('entry_exec_price', 0):.3f}</td>
            <td>{held_h:.1f}h</td>
            <td>{pos.get('fee_type', '—')}</td>
        </tr>"""
if not open_rows:
    open_rows = '<tr><td colspan="6" style="text-align:center;color:#999">No open positions</td></tr>'

# Recent closed trades (latest 25)
recent_rows = ""
for t in sorted(trades, key=lambda x: x.get("exit_ts", 0), reverse=True)[:25]:
    pnl = t.get("net_pnl_usd", 0)
    exit_ts = datetime.datetime.fromtimestamp(t.get("exit_ts", 0), datetime.timezone.utc).strftime("%m-%d %H:%M")
    recent_rows += f"""
        <tr>
            <td>{exit_ts}</td>
            <td>{escape(str(t.get('question', ''))[:55])}</td>
            <td>{t.get('entry_z', 0):+.1f}</td>
            <td>{'short' if t.get('direction') == -1 else 'long'}</td>
            <td>{t.get('hold_hours', 0):.1f}h</td>
            <td>{t.get('exit_reason', '—')}</td>
            <td style="color:{color_pnl(pnl)}">${pnl:+.2f}</td>
        </tr>"""
if not recent_rows:
    recent_rows = '<tr><td colspan="7" style="text-align:center;color:#999">No closed trades yet</td></tr>'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Polymarket Paper Trader · Dashboard</title>
<meta http-equiv="refresh" content="60">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
          max-width: 1200px; margin: 0 auto; padding: 20px; background: #f6f8fa; color: #1f2328; }}
  h1, h2 {{ color: #1f2328; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }}
  .subtitle {{ color: #656d76; font-size: 13px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; }}
  .card h3 {{ margin: 0; font-size: 12px; color: #656d76; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card .val {{ font-size: 28px; font-weight: 600; margin-top: 8px; }}
  .card .sub {{ font-size: 12px; color: #656d76; margin-top: 4px; }}
  .green {{ color: #2ea043; }}
  .red {{ color: #cf222e; }}
  .grey {{ color: #656d76; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #d0d7de; font-size: 13px; }}
  th {{ background: #f6f8fa; color: #656d76; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; }}
  tr:last-child td {{ border-bottom: none; }}
  .card-wide {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .compare {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 8px; font-size: 13px; }}
  .compare > div {{ padding: 8px; background: #f6f8fa; border-radius: 6px; }}
  .compare .label {{ color: #656d76; font-size: 11px; }}
  .compare .val {{ font-size: 16px; font-weight: 600; }}
</style>
</head>
<body>

<h1>Polymarket Paper Trader</h1>
<div class="subtitle">Last updated: {now_str} · Cycle #{cycles_run} · Started {started_at[:10] if started_at != '—' else '—'} · Auto-refresh 60s</div>

<div class="grid">
  <div class="card">
    <h3>Total Net PnL</h3>
    <div class="val {'green' if total_pnl > 0 else 'red' if total_pnl < 0 else 'grey'}">${total_pnl:+.2f}</div>
    <div class="sub">Across {closed_count} closed trades · ${total_fees:.2f} fees paid</div>
  </div>
  <div class="card">
    <h3>Win Rate</h3>
    <div class="val">{win_rate:.1f}%</div>
    <div class="sub">{wins} wins / {closed_count} trades · expected ~{EXP_WIN_RATE}%</div>
  </div>
  <div class="card">
    <h3>Avg Trade</h3>
    <div class="val {'green' if avg_trade > 0 else 'red' if avg_trade < 0 else 'grey'}">${avg_trade:+.2f}</div>
    <div class="sub">expected ~${EXP_AVG_TRADE:.2f}</div>
  </div>
  <div class="card">
    <h3>Open Positions</h3>
    <div class="val">{open_count}</div>
    <div class="sub">closed: {closed_count} · cycles: {cycles_run}</div>
  </div>
</div>

<div class="card-wide">
  <h2 style="margin-top:0">Equity curve</h2>
  <canvas id="equity"></canvas>
</div>

<div class="card-wide">
  <h2 style="margin-top:0">Per-trade PnL distribution</h2>
  <canvas id="hist"></canvas>
</div>

<div class="card-wide">
  <h2 style="margin-top:0">Backtest vs Live comparison</h2>
  <p style="color:#656d76;font-size:13px;margin:0 0 12px">Honest comparison to extended_v3 (20-month $5k-$50k volume) — the most realistic backtest population.</p>
  <div class="compare">
    <div>
      <div class="label">EXPECTED (backtest)</div>
      <div class="val">+${EXP_AVG_TRADE:.2f}/trade · {EXP_WIN_RATE}% win</div>
    </div>
    <div>
      <div class="label">LIVE (paper)</div>
      <div class="val">${avg_trade:+.2f}/trade · {win_rate:.1f}% win</div>
    </div>
    <div>
      <div class="label">DELTA</div>
      <div class="val">${avg_trade - EXP_AVG_TRADE:+.2f}/trade · {win_rate - EXP_WIN_RATE:+.1f} pp</div>
    </div>
  </div>
</div>

<div class="card-wide">
  <h2 style="margin-top:0">Open positions ({open_count})</h2>
  <table>
    <thead>
      <tr><th>Market</th><th>Entry z</th><th>Dir</th><th>Exec px</th><th>Held</th><th>Category</th></tr>
    </thead>
    <tbody>{open_rows}</tbody>
  </table>
</div>

<div class="card-wide">
  <h2 style="margin-top:0">Recent closed trades (last 25)</h2>
  <table>
    <thead>
      <tr><th>Exit time</th><th>Market</th><th>z</th><th>Dir</th><th>Held</th><th>Exit</th><th>PnL</th></tr>
    </thead>
    <tbody>{recent_rows}</tbody>
  </table>
</div>

<script>
const equityData = {{
    labels: {json.dumps(equity_x)},
    datasets: [{{
        label: 'Cumulative PnL ($)',
        data: {json.dumps(equity_y)},
        borderColor: '#0969da',
        backgroundColor: 'rgba(9, 105, 218, 0.1)',
        fill: true,
        tension: 0.1,
        pointRadius: 2,
    }}]
}};
new Chart(document.getElementById('equity'), {{
    type: 'line',
    data: equityData,
    options: {{ responsive: true, maintainAspectRatio: false,
              plugins: {{ legend: {{ display: false }} }},
              scales: {{ x: {{ ticks: {{ maxTicksLimit: 8 }} }} }} }}
}});

// PnL histogram
const pnls = {json.dumps(pnls)};
if (pnls.length > 0) {{
    const bins = 30;
    const min = Math.min(...pnls), max = Math.max(...pnls);
    const range = max - min || 1;
    const width = range / bins;
    const buckets = new Array(bins).fill(0);
    const labels = [];
    for (let i = 0; i < bins; i++) {{
        labels.push((min + width * (i + 0.5)).toFixed(1));
    }}
    pnls.forEach(p => {{
        let idx = Math.min(bins - 1, Math.floor((p - min) / width));
        buckets[idx]++;
    }});
    new Chart(document.getElementById('hist'), {{
        type: 'bar',
        data: {{ labels: labels, datasets: [{{
            label: 'Trade count', data: buckets,
            backgroundColor: labels.map(l => parseFloat(l) >= 0 ? '#2ea043' : '#cf222e')
        }}] }},
        options: {{ responsive: true, maintainAspectRatio: false,
                  plugins: {{ legend: {{ display: false }} }} }}
    }});
}} else {{
    document.getElementById('hist').parentElement.innerHTML += '<p style="color:#999;text-align:center">No trades yet</p>';
}}
document.getElementById('equity').style.height = '300px';
document.getElementById('hist').style.height = '250px';
</script>

</body>
</html>
"""

OUT.write_text(html)
print(f"Generated {OUT}")
print(f"  trades closed: {closed_count}")
print(f"  open positions: {open_count}")
print(f"  total PnL: ${total_pnl:+.2f}")
