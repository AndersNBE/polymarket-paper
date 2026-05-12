#!/usr/bin/env python3
"""
generate_dashboard.py — Build dashboard.html for paper trader.

Sections:
  - Header KPIs (PnL, win rate, avg trade, positions)
  - Equity curve with backtest baseline overlay
  - Drawdown chart
  - Rolling win-rate over last 20 trades
  - PnL histogram with mean/median markers
  - Z-score distribution at entry
  - Hold-time distribution
  - Per-category (feeType) breakdown
  - Open positions table
  - Recent closed trades
  - Live vs backtest comparison
  - Forecast cone (bootstrap projection of expected PnL)
"""
import json
import datetime
from pathlib import Path
from html import escape

HERE = Path(__file__).parent
STATE_FILE = HERE / "paper_state.json"
TRADES_FILE = HERE / "paper_trades.jsonl"
SIGNALS_FILE = HERE / "paper_signals.jsonl"
CYCLES_FILE = HERE / "paper_cycles.jsonl"
OUT = HERE / "dashboard.html"

# Backtest expectations (from extended_v3 — honest)
EXP_AVG_TRADE = 7.66
EXP_WIN_RATE = 54.3
EXP_STD = 70.79
EXP_TRADES_PER_DAY = 3.0   # middle scenario from monte_carlo_v3

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
cycles = load_jsonl(CYCLES_FILE)

now = datetime.datetime.now(datetime.timezone.utc)
now_str = now.strftime("%Y-%m-%d %H:%M UTC")
now_ts = now.timestamp()

# ── Core stats ───────────────────────────────────────────────────
closed = sorted(trades, key=lambda t: t.get("exit_ts", 0))
n_closed = len(closed)
open_positions = state.get("positions", {})
n_open = len(open_positions)
cycles_run = state.get("cycle", 0)
started_at = state.get("started_at", "—")

pnls = [t.get("net_pnl_usd", 0) for t in closed]
total_pnl = sum(pnls)
total_fees = sum(t.get("total_fees_usd", 0) for t in closed)
wins = sum(1 for p in pnls if p > 0)
win_rate = (wins / n_closed * 100) if n_closed else 0
avg_trade = (total_pnl / n_closed) if n_closed else 0

# CLV stats
clv_values = [t.get("clv_value") for t in closed if t.get("clv_value") is not None]
n_with_clv = len(clv_values)
avg_clv = (sum(clv_values) / n_with_clv) if n_with_clv else 0
clv_positive = sum(1 for c in clv_values if c > 0)
clv_winrate = (clv_positive / n_with_clv * 100) if n_with_clv else 0

# ── Equity curve + drawdown ──────────────────────────────────────
equity_x = []
equity_y = []
running = 0
peak = 0
drawdown_y = []
for t in closed:
    running += t.get("net_pnl_usd", 0)
    ts = t.get("exit_ts", 0)
    equity_x.append(datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"))
    equity_y.append(round(running, 2))
    peak = max(peak, running)
    drawdown_y.append(round(running - peak, 2))

max_dd = min(drawdown_y, default=0)

# ── Baseline: expected trajectory ────────────────────────────────
# If we'd hit backtest expectation, where would equity be?
baseline_y = [round(EXP_AVG_TRADE * (i + 1), 2) for i in range(len(equity_y))]

# ── Forecast cone (bootstrap from backtest distribution) ─────────
# Generate p5/p50/p95 projections for next 100 trades from current point
import random
random.seed(42)
# Use a normal-like distribution centered at expected
N_FORECAST = 100
N_SIM = 1000
projections_p5 = []
projections_p50 = []
projections_p95 = []
last_eq = equity_y[-1] if equity_y else 0
for k in range(1, N_FORECAST + 1):
    sims = []
    for _ in range(N_SIM):
        s = sum(random.gauss(EXP_AVG_TRADE, EXP_STD) for _ in range(k))
        sims.append(last_eq + s)
    sims.sort()
    projections_p5.append(round(sims[int(N_SIM * 0.05)], 2))
    projections_p50.append(round(sims[N_SIM // 2], 2))
    projections_p95.append(round(sims[int(N_SIM * 0.95)], 2))
projection_x = list(range(n_closed + 1, n_closed + 1 + N_FORECAST))

# ── Rolling win rate (window of 20) ──────────────────────────────
WINDOW = 20
rolling_winrate_x = []
rolling_winrate_y = []
for i in range(WINDOW - 1, n_closed):
    window = pnls[i - WINDOW + 1:i + 1]
    wr = sum(1 for p in window if p > 0) / WINDOW * 100
    rolling_winrate_x.append(i + 1)
    rolling_winrate_y.append(round(wr, 1))

# ── Z-score distribution at entry ────────────────────────────────
entry_zs = [abs(t.get("entry_z", 0)) for t in trades]

# ── Hold-time distribution ───────────────────────────────────────
hold_times = [t.get("hold_hours", 0) for t in trades]

# ── Per-category breakdown ───────────────────────────────────────
from collections import defaultdict
by_cat = defaultdict(list)
for t in closed:
    by_cat[t.get("fee_type") or "unknown"].append(t.get("net_pnl_usd", 0))

cat_rows = ""
for cat in sorted(by_cat.keys(), key=lambda k: -len(by_cat[k])):
    arr = by_cat[cat]
    cat_pnl = sum(arr)
    cat_wins = sum(1 for x in arr if x > 0)
    cat_winrate = cat_wins / len(arr) * 100
    cat_avg = sum(arr) / len(arr)
    color = "green" if cat_pnl > 0 else "red" if cat_pnl < 0 else "grey"
    cat_rows += f"""
        <tr>
            <td>{cat.replace('_fees', '').replace('_v2', '').replace('_prices', '')}</td>
            <td>{len(arr)}</td>
            <td>{cat_winrate:.0f}%</td>
            <td>${cat_avg:+.2f}</td>
            <td style="color:{'#2ea043' if cat_pnl > 0 else '#cf222e' if cat_pnl < 0 else '#666'}">${cat_pnl:+.2f}</td>
        </tr>"""
if not cat_rows:
    cat_rows = '<tr><td colspan="5" style="text-align:center;color:#999">No data yet</td></tr>'

# ── Tables ──────────────────────────────────────────────────────
def color_pnl(v):
    if v > 0: return "#2ea043"
    if v < 0: return "#cf222e"
    return "#666"

# Open positions with mark-to-market estimate (using current bid/ask if available)
open_rows = ""
for mid, pos in open_positions.items():
    held_h = (now_ts - pos.get("entry_ts", 0)) / 3600
    spread_bp = pos.get("entry_spread", 0) * 100  # in cents
    clv_text = f"{pos.get('clv_value', 0)*100:+.2f}¢" if pos.get("clv_value") is not None else "—"
    open_rows += f"""
        <tr>
            <td>{escape(str(pos.get('question', ''))[:60])}</td>
            <td>{pos.get('entry_z', 0):+.2f}</td>
            <td>{'short' if pos.get('direction') == -1 else 'long'}</td>
            <td>${pos.get('entry_exec_price', 0):.3f}</td>
            <td>{spread_bp:.1f}¢</td>
            <td>{pos.get('shares', 0):.1f}</td>
            <td>${pos.get('shares', 0) * pos.get('entry_exec_price', 0) if pos.get('direction')==1 else pos.get('shares', 0) * (1-pos.get('entry_exec_price', 0)):.1f}</td>
            <td>{held_h:.1f}h</td>
            <td>{pos.get('dte', 0):.0f}d</td>
            <td>{clv_text}</td>
            <td>{(pos.get('fee_type') or '—').replace('_fees', '').replace('_v2', '')}</td>
        </tr>"""
if not open_rows:
    open_rows = '<tr><td colspan="11" style="text-align:center;color:#999;padding:24px">No open positions yet</td></tr>'

# Recent closed trades
recent_rows = ""
for t in sorted(closed, key=lambda x: x.get("exit_ts", 0), reverse=True)[:30]:
    pnl = t.get("net_pnl_usd", 0)
    exit_dt = datetime.datetime.fromtimestamp(t.get("exit_ts", 0), datetime.timezone.utc).strftime("%m-%d %H:%M")
    entry_dt = datetime.datetime.fromtimestamp(t.get("entry_ts", 0), datetime.timezone.utc).strftime("%m-%d %H:%M")
    spread = t.get("entry_spread", 0) * 100
    stake = (t.get("shares", 0) * t.get("entry_exec_price", 0)) if t.get("direction") == 1 else (t.get("shares", 0) * (1 - t.get("entry_exec_price", 0)))
    clv_text = f"{t.get('clv_value', 0)*100:+.2f}¢" if t.get("clv_value") is not None else "—"
    fees = t.get("total_fees_usd", 0)
    recent_rows += f"""
        <tr>
            <td>{exit_dt}</td>
            <td>{escape(str(t.get('question', ''))[:45])}</td>
            <td>{t.get('entry_z', 0):+.1f}σ</td>
            <td>{'S' if t.get('direction') == -1 else 'L'}</td>
            <td>{t.get('entry_exec_price', 0):.3f}→{t.get('exit_exec_price', 0):.3f}</td>
            <td>{spread:.1f}¢</td>
            <td>${stake:.0f}</td>
            <td>{t.get('hold_hours', 0):.1f}h</td>
            <td>{t.get('exit_reason', '—')[:8]}</td>
            <td>{clv_text}</td>
            <td>${fees:.2f}</td>
            <td style="color:{color_pnl(pnl)};font-weight:600">${pnl:+.2f}</td>
        </tr>"""
if not recent_rows:
    recent_rows = '<tr><td colspan="12" style="text-align:center;color:#999;padding:24px">No closed trades yet — strategy needs z≥5 spikes to revert (hold up to 48h)</td></tr>'

# Best / worst trade
best = max(closed, key=lambda x: x.get("net_pnl_usd", 0)) if closed else None
worst = min(closed, key=lambda x: x.get("net_pnl_usd", 0)) if closed else None

best_html = f"<div><strong>Best:</strong> ${best.get('net_pnl_usd', 0):+.2f} · {escape(str(best.get('question', ''))[:50])} · z={best.get('entry_z', 0):+.1f}</div>" if best else ""
worst_html = f"<div><strong>Worst:</strong> ${worst.get('net_pnl_usd', 0):+.2f} · {escape(str(worst.get('question', ''))[:50])} · z={worst.get('entry_z', 0):+.1f}</div>" if worst else ""

# Signals seen but not yet closed
n_signals_total = len(signals)
n_signals_filtered = n_closed + n_open  # rough

# Cycle activity (heartbeat / health)
recent_cycles = sorted(cycles, key=lambda c: c.get("ts", 0), reverse=True)[:30]
cycle_rows = ""
for c in recent_cycles:
    ts = c.get("ts", 0)
    dt_str = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%m-%d %H:%M")
    rej = c.get("rejected", {}) or {}
    main_rej = "  ".join([f"{k}={v}" for k, v in rej.items() if v > 0][:3])  # top 3 rejection reasons
    cycle_rows += f"""
        <tr>
            <td>{dt_str}</td>
            <td>#{c.get('cycle', 0)}</td>
            <td>{c.get('universe_size', 0):,}</td>
            <td>{c.get('scanned', 0):,}</td>
            <td>{c.get('signals_at_z5', 0)}</td>
            <td>{c.get('opened', 0)}</td>
            <td>{c.get('closed', 0)}</td>
            <td>{c.get('open_positions_after', 0)}</td>
            <td style="font-size:10px;color:#666">{main_rej}</td>
        </tr>"""
if not cycle_rows:
    cycle_rows = '<tr><td colspan="9" style="text-align:center;color:#999;padding:24px">No cycle data yet</td></tr>'

# Lifetime totals
total_markets_scanned = sum(c.get("scanned", 0) for c in cycles)
total_signals_seen = sum(c.get("signals_at_z5", 0) for c in cycles)
last_cycle_ts = max((c.get("ts", 0) for c in cycles), default=0)
mins_since_last = (now_ts - last_cycle_ts) / 60 if last_cycle_ts else None
liveness = "🟢 Active" if mins_since_last is not None and mins_since_last < 30 else "🟡 Stale" if mins_since_last is not None and mins_since_last < 120 else "🔴 Dead"
liveness_text = f"{liveness}  (last cycle {mins_since_last:.0f} min ago)" if mins_since_last is not None else "Waiting for first cycle"

# ── HTML ─────────────────────────────────────────────────────────
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
  h1, h2 {{ color: #1f2328; margin: 0; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; margin-bottom: 8px; }}
  .subtitle {{ color: #656d76; font-size: 12px; margin-bottom: 20px; }}
  .grid4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }}
  .grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 16px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px; }}
  .card {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; }}
  .card h3 {{ margin: 0; font-size: 11px; color: #656d76; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card .val {{ font-size: 24px; font-weight: 600; margin-top: 6px; }}
  .card .sub {{ font-size: 11px; color: #656d76; margin-top: 2px; }}
  .green {{ color: #2ea043; }}
  .red {{ color: #cf222e; }}
  .grey {{ color: #656d76; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; font-size: 12px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #d0d7de; }}
  th {{ background: #f6f8fa; color: #656d76; text-transform: uppercase; font-size: 10px; letter-spacing: 0.05em; }}
  tr:last-child td {{ border-bottom: none; }}
  td:last-child, th:last-child {{ text-align: right; }}
  .card-wide {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; margin-bottom: 12px; }}
  canvas {{ max-height: 280px; }}
  .empty-msg {{ color: #999; text-align: center; padding: 30px; font-size: 13px; }}
  .chart-cell {{ position: relative; height: 220px; }}
  .highlight {{ background: #fff8c5; padding: 8px 12px; border-radius: 6px; font-size: 12px; margin-top: 8px; }}
  .legend-row {{ display: flex; gap: 16px; font-size: 11px; color: #656d76; margin-top: 4px; }}
  .legend-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>

<h1>📊 Polymarket Paper Trader</h1>
<div class="subtitle">Updated {now_str} · Cycle #{cycles_run} · Started {started_at[:10] if started_at != '—' else '—'} · Auto-refresh 60s · {liveness_text}</div>

<div class="grid4">
  <div class="card">
    <h3>Net PnL</h3>
    <div class="val {'green' if total_pnl > 0 else 'red' if total_pnl < 0 else 'grey'}">${total_pnl:+.2f}</div>
    <div class="sub">{n_closed} closed · ${total_fees:.2f} fees paid</div>
  </div>
  <div class="card">
    <h3>Win Rate</h3>
    <div class="val">{win_rate:.1f}%</div>
    <div class="sub">{wins}/{n_closed} · expected ~{EXP_WIN_RATE}%</div>
  </div>
  <div class="card">
    <h3>Avg / Trade</h3>
    <div class="val {'green' if avg_trade > 0 else 'red' if avg_trade < 0 else 'grey'}">${avg_trade:+.2f}</div>
    <div class="sub">expected ~${EXP_AVG_TRADE:.2f}</div>
  </div>
  <div class="card">
    <h3>Positions</h3>
    <div class="val">{n_open} <span style="font-size:14px;color:#666">open</span></div>
    <div class="sub">{n_closed} closed · {cycles_run} scans</div>
  </div>
</div>

<div class="grid4">
  <div class="card">
    <h3>Max Drawdown</h3>
    <div class="val {'red' if max_dd < 0 else 'grey'}">${max_dd:+.2f}</div>
    <div class="sub">worst peak-to-trough</div>
  </div>
  <div class="card">
    <h3>Avg CLV (closing line value)</h3>
    <div class="val {'green' if avg_clv > 0 else 'red' if avg_clv < 0 else 'grey'}">{avg_clv*100:+.2f}¢</div>
    <div class="sub">{n_with_clv} measured · {clv_winrate:.0f}% positive · ★ best edge proxy</div>
  </div>
  <div class="card">
    <h3>Total Signals Seen</h3>
    <div class="val">{n_signals_total}</div>
    <div class="sub">passed all filters</div>
  </div>
  <div class="card">
    <h3>Strategy Status</h3>
    <div class="val" style="font-size:14px;color:#0969da">{'✓ Matching backtest' if abs(avg_trade - EXP_AVG_TRADE) < 5 and n_closed > 10 else '⏳ Gathering data' if n_closed < 10 else '⚠ Deviation from backtest'}</div>
    <div class="sub">{'expectation' if n_closed > 10 else 'need 10+ trades for assessment'}</div>
  </div>
</div>

<div class="card-wide">
  <h2>📈 Equity curve (live vs expected)</h2>
  <div class="legend-row">
    <div><span class="legend-dot" style="background:#0969da"></span>Live PnL</div>
    <div><span class="legend-dot" style="background:#999"></span>Backtest expectation (avg $7.66/trade)</div>
    <div><span class="legend-dot" style="background:#cf222e"></span>Break-even</div>
  </div>
  <div class="chart-cell"><canvas id="equity"></canvas></div>
  {f'<div class="empty-msg">Waiting for first closed trade — strategy needs z≥5 spikes to revert (typically 0-2 per cycle)</div>' if not closed else ''}
</div>

<div class="grid2">
  <div class="card-wide">
    <h2>📉 Drawdown</h2>
    <div class="chart-cell"><canvas id="drawdown"></canvas></div>
    {f'<div class="empty-msg">No drawdown to plot yet</div>' if not closed else ''}
  </div>
  <div class="card-wide">
    <h2>🎯 Rolling 20-trade win rate</h2>
    <div class="chart-cell"><canvas id="rollwin"></canvas></div>
    {f'<div class="empty-msg">Need 20+ trades for rolling stat</div>' if len(rolling_winrate_x) == 0 else ''}
  </div>
</div>

<div class="card-wide">
  <h2>🔮 Forecast cone (bootstrap of next 100 trades, based on backtest)</h2>
  <div class="legend-row">
    <div><span class="legend-dot" style="background:#0969da"></span>Live equity</div>
    <div><span class="legend-dot" style="background:rgba(46,160,67,0.3)"></span>5%-95% projection cone</div>
    <div><span class="legend-dot" style="background:#2ea043"></span>Median projection</div>
  </div>
  <div class="chart-cell" style="height:260px"><canvas id="forecast"></canvas></div>
  <div class="highlight">
    With $7.66 expected/trade and $70 std (from backtest), bootstrap says:
    in 100 more trades, projected PnL median = +${EXP_AVG_TRADE * 100:.0f},
    p5 = ${EXP_AVG_TRADE * 100 - 1.65 * EXP_STD * (100**0.5):.0f},
    p95 = ${EXP_AVG_TRADE * 100 + 1.65 * EXP_STD * (100**0.5):.0f}.
  </div>
</div>

<div class="grid2">
  <div class="card-wide">
    <h2>💰 PnL distribution</h2>
    <div class="chart-cell"><canvas id="pnlhist"></canvas></div>
    {f'<div class="empty-msg">No trades yet</div>' if not closed else ''}
  </div>
  <div class="card-wide">
    <h2>⏱ Hold-time distribution</h2>
    <div class="chart-cell"><canvas id="holdhist"></canvas></div>
    {f'<div class="empty-msg">No trades yet</div>' if not trades else ''}
  </div>
</div>

<div class="grid2">
  <div class="card-wide">
    <h2>⚡ Entry z-score distribution</h2>
    <div class="chart-cell"><canvas id="zhist"></canvas></div>
    {f'<div class="empty-msg">No signals yet</div>' if not trades else ''}
  </div>
  <div class="card-wide">
    <h2>🏷 Performance by market category</h2>
    <table>
      <thead>
        <tr><th>Category</th><th>n</th><th>Win%</th><th>Avg</th><th>Total</th></tr>
      </thead>
      <tbody>{cat_rows}</tbody>
    </table>
  </div>
</div>

<div class="card-wide">
  <h2>🫀 Bot health — recent cycles (last {len(recent_cycles)})</h2>
  <p style="color:#656d76;font-size:12px;margin:0 0 8px">Each cycle scans ~3000 markets. Most are filtered out. The bot is healthy if cycles keep appearing here.</p>
  <table>
    <thead>
      <tr><th>Time</th><th>#</th><th>Universe</th><th>Scanned</th><th>z≥5 signals</th><th>Opened</th><th>Closed</th><th>Now open</th><th>Top filter rejections</th></tr>
    </thead>
    <tbody>{cycle_rows}</tbody>
  </table>
  <div class="legend-row" style="margin-top:8px">
    <div>Lifetime: <strong>{total_markets_scanned:,}</strong> markets scanned · <strong>{total_signals_seen}</strong> z≥5 signals seen · <strong>{n_open}</strong> currently open</div>
  </div>
</div>

<div class="card-wide">
  <h2>🎮 Open positions ({n_open})</h2>
  <table>
    <thead>
      <tr>
        <th>Market</th>
        <th>z</th>
        <th>Dir</th>
        <th>Exec px</th>
        <th>Spread</th>
        <th>Shares</th>
        <th>Stake</th>
        <th>Held</th>
        <th>DTE</th>
        <th>CLV</th>
        <th>Category</th>
      </tr>
    </thead>
    <tbody>{open_rows}</tbody>
  </table>
</div>

<div class="card-wide">
  <h2>📋 Recent closed trades (latest 30)</h2>
  {best_html}
  {worst_html}
  <table style="margin-top:8px">
    <thead>
      <tr>
        <th>Exit time</th>
        <th>Market</th>
        <th>z</th>
        <th>Dir</th>
        <th>Px entry→exit</th>
        <th>Spread</th>
        <th>Stake</th>
        <th>Held</th>
        <th>Exit why</th>
        <th>CLV</th>
        <th>Fees</th>
        <th>Net PnL</th>
      </tr>
    </thead>
    <tbody>{recent_rows}</tbody>
  </table>
</div>

<script>
const COMMON = {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ mode: 'index', intersect: false }} }},
    interaction: {{ mode: 'nearest', intersect: false }}
}};

const equity_x = {json.dumps(equity_x)};
const equity_y = {json.dumps(equity_y)};
const baseline_y = {json.dumps(baseline_y)};

if (equity_y.length > 0) {{
    new Chart(document.getElementById('equity'), {{
        type: 'line',
        data: {{
            labels: equity_x,
            datasets: [
                {{ label: 'Live PnL', data: equity_y, borderColor: '#0969da', backgroundColor: 'rgba(9, 105, 218, 0.1)', fill: true, tension: 0.1, pointRadius: 2 }},
                {{ label: 'Expected (backtest)', data: baseline_y, borderColor: '#999', borderDash: [4, 4], pointRadius: 0, fill: false }},
                {{ label: 'Break-even', data: equity_y.map(() => 0), borderColor: '#cf222e', borderDash: [2, 2], pointRadius: 0, fill: false }},
            ]
        }},
        options: {{ ...COMMON, scales: {{ x: {{ ticks: {{ maxTicksLimit: 8 }} }} }} }}
    }});
}}

const drawdown_y = {json.dumps(drawdown_y)};
if (drawdown_y.length > 0) {{
    new Chart(document.getElementById('drawdown'), {{
        type: 'line',
        data: {{ labels: equity_x, datasets: [{{ data: drawdown_y, borderColor: '#cf222e', backgroundColor: 'rgba(207, 34, 46, 0.15)', fill: 'origin', pointRadius: 0 }}] }},
        options: {{ ...COMMON, scales: {{ x: {{ ticks: {{ maxTicksLimit: 6 }} }}, y: {{ max: 0 }} }} }}
    }});
}}

const rw_x = {json.dumps(rolling_winrate_x)};
const rw_y = {json.dumps(rolling_winrate_y)};
if (rw_y.length > 0) {{
    new Chart(document.getElementById('rollwin'), {{
        type: 'line',
        data: {{ labels: rw_x, datasets: [
            {{ data: rw_y, borderColor: '#2ea043', backgroundColor: 'rgba(46, 160, 67, 0.1)', fill: true, pointRadius: 2 }},
            {{ data: rw_y.map(() => {EXP_WIN_RATE}), borderColor: '#999', borderDash: [4, 4], pointRadius: 0, fill: false, label: 'expected' }}
        ] }},
        options: {{ ...COMMON, scales: {{ y: {{ min: 0, max: 100 }} }} }}
    }});
}}

// Forecast cone
const proj_x = {json.dumps(projection_x)};
const proj_p5 = {json.dumps(projections_p5)};
const proj_p50 = {json.dumps(projections_p50)};
const proj_p95 = {json.dumps(projections_p95)};
const forecast_labels = equity_y.concat(proj_p50.map(() => '')).map((_, i) => i + 1);
const live_padded = equity_y.concat(proj_p50.map(() => null));
const p5_padded = equity_y.map(() => null).concat(proj_p5);
const p50_padded = equity_y.map(() => null).concat(proj_p50);
const p95_padded = equity_y.map(() => null).concat(proj_p95);
new Chart(document.getElementById('forecast'), {{
    type: 'line',
    data: {{ labels: forecast_labels, datasets: [
        {{ label: 'p95', data: p95_padded, borderColor: 'rgba(46,160,67,0.5)', backgroundColor: 'rgba(46,160,67,0.15)', fill: '+1', pointRadius: 0 }},
        {{ label: 'p5', data: p5_padded, borderColor: 'rgba(46,160,67,0.5)', backgroundColor: 'rgba(46,160,67,0.15)', fill: false, pointRadius: 0 }},
        {{ label: 'median', data: p50_padded, borderColor: '#2ea043', pointRadius: 0, fill: false }},
        {{ label: 'Live', data: live_padded, borderColor: '#0969da', backgroundColor: 'rgba(9,105,218,0.2)', pointRadius: 2, fill: false }},
    ] }},
    options: {{ ...COMMON, scales: {{ x: {{ title: {{ display: true, text: 'Trade #' }} }} }} }}
}});

// PnL histogram
const pnls = {json.dumps(pnls)};
if (pnls.length > 0) {{
    const bins = 25; const min = Math.min(...pnls), max = Math.max(...pnls);
    const range = max - min || 1; const w = range / bins;
    const buckets = new Array(bins).fill(0);
    const labels = [];
    for (let i = 0; i < bins; i++) labels.push((min + w * (i + 0.5)).toFixed(1));
    pnls.forEach(p => buckets[Math.min(bins - 1, Math.floor((p - min) / w))]++);
    new Chart(document.getElementById('pnlhist'), {{
        type: 'bar',
        data: {{ labels: labels, datasets: [{{ data: buckets, backgroundColor: labels.map(l => parseFloat(l) >= 0 ? '#2ea043' : '#cf222e') }}] }},
        options: COMMON
    }});
}}

// Hold-time histogram
const holds = {json.dumps(hold_times)};
if (holds.length > 0) {{
    const bins = 12; const max = Math.max(...holds, 48); const w = max / bins;
    const buckets = new Array(bins).fill(0); const labels = [];
    for (let i = 0; i < bins; i++) labels.push((w * (i + 0.5)).toFixed(0) + 'h');
    holds.forEach(h => buckets[Math.min(bins - 1, Math.floor(h / w))]++);
    new Chart(document.getElementById('holdhist'), {{
        type: 'bar',
        data: {{ labels: labels, datasets: [{{ data: buckets, backgroundColor: '#0969da' }}] }},
        options: COMMON
    }});
}}

// Z-score histogram
const zs = {json.dumps(entry_zs)};
if (zs.length > 0) {{
    const bins = 15; const min = Math.min(...zs), max = Math.max(...zs);
    const range = max - min || 1; const w = range / bins;
    const buckets = new Array(bins).fill(0); const labels = [];
    for (let i = 0; i < bins; i++) labels.push((min + w * (i + 0.5)).toFixed(1));
    zs.forEach(z => buckets[Math.min(bins - 1, Math.floor((z - min) / w))]++);
    new Chart(document.getElementById('zhist'), {{
        type: 'bar',
        data: {{ labels: labels, datasets: [{{ data: buckets, backgroundColor: '#a371f7' }}] }},
        options: COMMON
    }});
}}
</script>

</body>
</html>
"""

OUT.write_text(html)
print(f"Generated {OUT}")
print(f"  closed trades: {n_closed}")
print(f"  open positions: {n_open}")
print(f"  total PnL: ${total_pnl:+.2f}")
