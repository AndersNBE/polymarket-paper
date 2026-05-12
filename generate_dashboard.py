#!/usr/bin/env python3
"""
generate_dashboard.py — Pro trading dashboard for paper trader.

Dark mode, monospace numbers, live indicators, narrative-driven layout.
Regenerated on every workflow cycle; served via GitHub Pages.
"""
import json
import datetime
import random
from pathlib import Path
from html import escape
from collections import defaultdict

HERE = Path(__file__).parent
STATE_FILE = HERE / "paper_state.json"
TRADES_FILE = HERE / "paper_trades.jsonl"
SIGNALS_FILE = HERE / "paper_signals.jsonl"
CYCLES_FILE = HERE / "paper_cycles.jsonl"
OUT = HERE / "dashboard.html"

# Backtest expectations (from extended_v3)
EXP_AVG_TRADE = 7.66
EXP_WIN_RATE = 54.3
EXP_STD = 70.79
EXP_CLV = 2.42       # cents per share, our gold metric
EXP_TRADES_PER_DAY = 3.0
BANKROLL_USD = 720
STAKE = 30

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
now_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
now_ts = now.timestamp()

# ── Stats ────────────────────────────────────────────────────────
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

clv_values = [t.get("clv_value") for t in closed if t.get("clv_value") is not None]
n_with_clv = len(clv_values)
avg_clv = (sum(clv_values) / n_with_clv) if n_with_clv else 0
clv_positive = sum(1 for c in clv_values if c > 0)
clv_winrate = (clv_positive / n_with_clv * 100) if n_with_clv else 0

# Equity + drawdown
equity_x = []
equity_y = []
running = 0
peak = 0
drawdown_y = []
for t in closed:
    running += t.get("net_pnl_usd", 0)
    ts = t.get("exit_ts", 0)
    equity_x.append(datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%m-%d %H:%M"))
    equity_y.append(round(running, 2))
    peak = max(peak, running)
    drawdown_y.append(round(running - peak, 2))
max_dd = min(drawdown_y, default=0)

baseline_y = [round(EXP_AVG_TRADE * (i + 1) * (STAKE / 100), 2) for i in range(len(equity_y))]

# Forecast cone
random.seed(42)
N_FORECAST = 100
N_SIM = 1000
projections_p5 = []
projections_p50 = []
projections_p95 = []
last_eq = equity_y[-1] if equity_y else 0
exp_per_trade_at_stake = EXP_AVG_TRADE * (STAKE / 100)
exp_std_at_stake = EXP_STD * (STAKE / 100)
for k in range(1, N_FORECAST + 1):
    sims = sorted(last_eq + sum(random.gauss(exp_per_trade_at_stake, exp_std_at_stake) for _ in range(k)) for _ in range(N_SIM))
    projections_p5.append(round(sims[int(N_SIM * 0.05)], 2))
    projections_p50.append(round(sims[N_SIM // 2], 2))
    projections_p95.append(round(sims[int(N_SIM * 0.95)], 2))

# Rolling win rate
WINDOW = 20
rolling_winrate_x = []
rolling_winrate_y = []
for i in range(WINDOW - 1, n_closed):
    window = pnls[i - WINDOW + 1:i + 1]
    wr = sum(1 for p in window if p > 0) / WINDOW * 100
    rolling_winrate_x.append(i + 1)
    rolling_winrate_y.append(round(wr, 1))

# Per-category breakdown
by_cat = defaultdict(list)
for t in closed:
    by_cat[t.get("fee_type") or "unknown"].append(t.get("net_pnl_usd", 0))

# Cycle activity
recent_cycles = sorted(cycles, key=lambda c: c.get("ts", 0), reverse=True)[:24]
total_markets_scanned = sum(c.get("scanned", 0) for c in cycles)
total_signals_seen = sum(c.get("signals_at_z5", 0) for c in cycles)
last_cycle_ts = max((c.get("ts", 0) for c in cycles), default=0)
mins_since_last = (now_ts - last_cycle_ts) / 60 if last_cycle_ts else None
if mins_since_last is None:
    liveness = "—"
    liveness_color = "#7d8590"
elif mins_since_last < 30:
    liveness = "LIVE"
    liveness_color = "#3fb950"
elif mins_since_last < 120:
    liveness = "STALE"
    liveness_color = "#d29922"
else:
    liveness = "DEAD"
    liveness_color = "#f85149"

# Top filter rejection reasons (lifetime)
lifetime_rejections = defaultdict(int)
for c in cycles:
    for k, v in (c.get("rejected") or {}).items():
        lifetime_rejections[k] += v

# ── Streak analysis ─────────────────────────────────────────────
current_streak = 0
current_streak_type = "neutral"
if closed:
    last_sign = 1 if closed[-1].get("net_pnl_usd", 0) > 0 else (-1 if closed[-1].get("net_pnl_usd", 0) < 0 else 0)
    if last_sign != 0:
        for t in reversed(closed):
            sign = 1 if t.get("net_pnl_usd", 0) > 0 else (-1 if t.get("net_pnl_usd", 0) < 0 else 0)
            if sign != last_sign or sign == 0: break
            current_streak += 1
        current_streak_type = "win" if last_sign > 0 else "loss"

# Best/worst single trade
best_trade = max(closed, key=lambda t: t.get("net_pnl_usd", 0)) if closed else None
worst_trade = min(closed, key=lambda t: t.get("net_pnl_usd", 0)) if closed else None
best_pnl = best_trade.get("net_pnl_usd", 0) if best_trade else 0
worst_pnl = worst_trade.get("net_pnl_usd", 0) if worst_trade else 0
best_q = escape(str(best_trade.get("question", "—"))[:50]) if best_trade else "no trades yet"
worst_q = escape(str(worst_trade.get("question", "—"))[:50]) if worst_trade else "no trades yet"

# Calendar heatmap — PnL by day
daily_pnl = defaultdict(float)
daily_count = defaultdict(int)
for t in closed:
    d = datetime.datetime.fromtimestamp(t.get("exit_ts", 0), datetime.timezone.utc).strftime("%Y-%m-%d")
    daily_pnl[d] += t.get("net_pnl_usd", 0)
    daily_count[d] += 1

# Calendar grid: last 21 days
cal_dates = [(now - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(20, -1, -1)]
cal_cells_html = ""
max_abs_daily = max((abs(daily_pnl[d]) for d in cal_dates), default=1) or 1
for d in cal_dates:
    pnl = daily_pnl.get(d, 0)
    n = daily_count.get(d, 0)
    if n == 0:
        bg = "#161b22"
        title = f"{d}: no trades"
        txt_color = "#6e7681"
    else:
        intensity = min(1.0, abs(pnl) / max_abs_daily)
        if pnl > 0:
            bg = f"rgba(63, 185, 80, {0.2 + 0.7*intensity})"
        elif pnl < 0:
            bg = f"rgba(248, 81, 73, {0.2 + 0.7*intensity})"
        else:
            bg = "#21262d"
        txt_color = "#e6edf3"
        title = f"{d}: ${pnl:+.2f} ({n} trades)"
    weekday = datetime.datetime.strptime(d, "%Y-%m-%d").strftime("%a")[:1]
    day_num = datetime.datetime.strptime(d, "%Y-%m-%d").strftime("%d")
    cal_cells_html += f'<div class="cal-cell" style="background:{bg};color:{txt_color}" title="{title}"><span class="cal-day">{day_num}</span><span class="cal-pnl">{f"${pnl:+.0f}" if n>0 else "·"}</span></div>'

# Waterfall: gross → spread → fees → gas → net
total_gross = sum(t.get("gross_pnl_usd", 0) for t in closed)
total_fees_only = sum((t.get("entry_fee", 0) + t.get("exit_fee", 0)) for t in closed)
total_gas = sum((t.get("entry_gas", 0) + t.get("exit_gas", 0)) for t in closed)
total_spread_est = total_fees - total_fees_only - total_gas  # rough; we lump spread into fees in code
total_net = sum(t.get("net_pnl_usd", 0) for t in closed)

# Next signal estimate
if cycles and len(cycles) > 3:
    recent_signals = sum(c.get("signals_at_z5", 0) for c in cycles[-12:])
    recent_hours = (cycles[-1]["ts"] - cycles[-12]["ts"]) / 3600 if len(cycles) >= 12 else 1
    sig_per_hr = recent_signals / recent_hours if recent_hours > 0 else 0
    next_signal_est = f"~{60/sig_per_hr:.0f}min" if sig_per_hr > 0 else "uncertain"
else:
    next_signal_est = "—"

# Capital utilization
total_deployed = sum(p.get("shares", 0) * (p.get("entry_exec_price", 0) if p.get("direction") == 1 else (1 - p.get("entry_exec_price", 0))) for p in open_positions.values())
util_pct = 100 * total_deployed / BANKROLL_USD if BANKROLL_USD else 0

# Sparklines (last 20 datapoints per metric)
spark_equity = equity_y[-20:] if len(equity_y) >= 2 else []

spark_clv = []
running_clv = 0
clv_running_count = 0
for t in closed:
    if t.get("clv_value") is not None:
        clv_running_count += 1
        running_clv += t["clv_value"]
        spark_clv.append(round(100 * running_clv / clv_running_count, 3))  # cumulative avg CLV in cents
spark_clv = spark_clv[-20:]

spark_winrate = []
if len(pnls) >= 5:
    for i in range(min(20, len(pnls))):
        # Compute win rate up to that point
        end_idx = len(pnls) - 20 + i + 1
        if end_idx < 5: continue
        chunk = pnls[:end_idx]
        spark_winrate.append(round(100 * sum(1 for x in chunk if x > 0) / len(chunk), 1))

spark_positions = []
for c in sorted(cycles, key=lambda c: c.get("ts", 0))[-20:]:
    spark_positions.append(c.get("open_positions_after", 0))

# Trade tape — last 30 trades as binary outcomes
tape_data = []
for t in closed[-30:]:
    pnl = t.get("net_pnl_usd", 0)
    tape_data.append({
        "pnl": round(pnl, 2),
        "win": pnl > 0,
        "z": round(t.get("entry_z", 0), 1),
        "q": str(t.get("question", ""))[:40],
    })
tape_html = ""
for td in tape_data:
    color = "#3fb950" if td["win"] else "#f85149" if td["pnl"] < 0 else "#7d8590"
    tooltip = f"{td['q']} | z={td['z']:+} | ${td['pnl']:+.2f}"
    tape_html += f'<div class="tape-dot" style="background:{color}" title="{escape(tooltip)}"></div>'
if not tape_html:
    tape_html = '<div class="empty-state-small" style="padding:8px">No trades yet</div>'

# Time-of-day heatmap: avg PnL per (weekday, hour) bucket
hod_buckets = defaultdict(lambda: {"pnl": 0.0, "count": 0})
for t in closed:
    dt = datetime.datetime.fromtimestamp(t.get("entry_ts", 0), datetime.timezone.utc)
    key = (dt.weekday(), dt.hour)  # 0=Mon
    hod_buckets[key]["pnl"] += t.get("net_pnl_usd", 0)
    hod_buckets[key]["count"] += 1

# Build heatmap HTML (rows = days Mon-Sun, columns = hours 0-23)
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
max_abs_cell = max((abs(v["pnl"]) for v in hod_buckets.values()), default=1) or 1
hod_html = '<div class="hod-grid">'
hod_html += '<div class="hod-corner"></div>'
for h in range(24):
    label = f"{h:02d}" if h % 3 == 0 else ""
    hod_html += f'<div class="hod-hour-label">{label}</div>'
for d in range(7):
    hod_html += f'<div class="hod-day-label">{day_labels[d]}</div>'
    for h in range(24):
        cell = hod_buckets.get((d, h))
        if cell and cell["count"] > 0:
            pnl = cell["pnl"]
            intensity = min(1.0, abs(pnl) / max_abs_cell)
            if pnl > 0:
                bg = f"rgba(63, 185, 80, {0.2 + 0.7*intensity})"
            else:
                bg = f"rgba(248, 81, 73, {0.2 + 0.7*intensity})"
            tooltip = f"{day_labels[d]} {h:02d}:00  ${pnl:+.2f} ({cell['count']} trades)"
        else:
            bg = "#161b22"
            tooltip = f"{day_labels[d]} {h:02d}:00  no trades"
        hod_html += f'<div class="hod-cell" style="background:{bg}" title="{tooltip}"></div>'
hod_html += '</div>'

# Portfolio donut: open positions by category
portfolio_cats = defaultdict(float)
for pos in open_positions.values():
    cat = (pos.get("fee_type") or "other").replace("_fees", "").replace("_v2", "").replace("_prices", "")
    stake = pos.get("shares", 0) * (pos.get("entry_exec_price", 0) if pos.get("direction") == 1 else (1 - pos.get("entry_exec_price", 0)))
    portfolio_cats[cat] += stake
portfolio_labels = list(portfolio_cats.keys())
portfolio_values = [round(v, 2) for v in portfolio_cats.values()]

# Streak history — walk all trades, identify each streak
streak_history = []
if closed:
    cur_sign = None
    cur_len = 0
    for t in closed:
        sign = 1 if t.get("net_pnl_usd", 0) > 0 else -1 if t.get("net_pnl_usd", 0) < 0 else 0
        if sign == cur_sign:
            cur_len += 1
        else:
            if cur_sign is not None and cur_sign != 0:
                streak_history.append(cur_len * cur_sign)
            cur_sign = sign
            cur_len = 1 if sign != 0 else 0
    if cur_sign is not None and cur_sign != 0:
        streak_history.append(cur_len * cur_sign)
streak_history = streak_history[-30:]   # last 30 streaks

# Sparklines for hero (use recent equity if available)
sparkline_data = equity_y[-30:] if equity_y else []

# Decision verdict
def get_verdict():
    if n_closed < 5:
        return ("INSUFFICIENT DATA", "#d29922", f"Need {5 - n_closed} more trades for first read")
    if n_with_clv < 5:
        return ("CLV PENDING", "#d29922", f"CLV measured on {n_with_clv} trades; need 10+ for confidence")
    if avg_clv > 0.005 and avg_trade > 0:
        return ("EDGE CONFIRMED", "#3fb950", "Both CLV and PnL positive — strategy working as expected")
    if avg_clv > 0.005 and avg_trade <= 0:
        return ("EDGE OK, VARIANCE", "#d29922", "CLV positive but PnL noisy — wait for more trades")
    if avg_clv <= 0 and avg_trade > 0:
        return ("LUCKY", "#d29922", "PnL positive but CLV not — may be variance, not edge")
    return ("NO EDGE DETECTED", "#f85149", "Both CLV and PnL negative — strategy likely not working")

verdict_label, verdict_color, verdict_sub = get_verdict()

# ── Helpers ──────────────────────────────────────────────────────
def fmt_pnl(v, prec=2):
    sign = "+" if v >= 0 else ""
    color = "#3fb950" if v > 0 else "#f85149" if v < 0 else "#7d8590"
    return f'<span style="color:{color}">{sign}${v:.{prec}f}</span>'

def color_for(v):
    if v > 0: return "#3fb950"
    if v < 0: return "#f85149"
    return "#7d8590"

# ── Build position cards (instead of table) ─────────────────────
position_cards = ""
for mid, pos in sorted(open_positions.items(), key=lambda kv: kv[1].get("entry_ts", 0), reverse=True):
    held_h = (now_ts - pos.get("entry_ts", 0)) / 3600
    entry_px = pos.get("entry_exec_price", 0)
    stake = pos.get("shares", 0) * (entry_px if pos.get("direction") == 1 else (1 - entry_px))
    dir_label = "SHORT" if pos.get("direction") == -1 else "LONG"
    dir_color = "#f85149" if pos.get("direction") == -1 else "#3fb950"
    clv = pos.get("clv_value")
    clv_html = f'<div class="pos-stat"><span class="pos-stat-label">CLV</span><span class="pos-stat-val" style="color:{color_for(clv)}">{clv*100:+.2f}¢</span></div>' if clv is not None else '<div class="pos-stat"><span class="pos-stat-label">CLV</span><span class="pos-stat-val" style="color:#7d8590">pending</span></div>'
    spread = pos.get("entry_spread", 0) * 100
    category = (pos.get('fee_type') or 'other').replace('_fees', '').replace('_v2', '').replace('_prices', '')
    position_cards += f"""
    <div class="pos-card">
        <div class="pos-card-top">
            <div class="pos-question">{escape(str(pos.get('question', ''))[:80])}</div>
            <div class="pos-dir" style="color:{dir_color}">{dir_label}</div>
        </div>
        <div class="pos-stats">
            <div class="pos-stat"><span class="pos-stat-label">z-score</span><span class="pos-stat-val">{pos.get('entry_z', 0):+.2f}σ</span></div>
            <div class="pos-stat"><span class="pos-stat-label">Entry</span><span class="pos-stat-val">${entry_px:.3f}</span></div>
            <div class="pos-stat"><span class="pos-stat-label">Spread</span><span class="pos-stat-val">{spread:.1f}¢</span></div>
            <div class="pos-stat"><span class="pos-stat-label">Stake</span><span class="pos-stat-val">${stake:.0f}</span></div>
            <div class="pos-stat"><span class="pos-stat-label">Held</span><span class="pos-stat-val">{held_h:.1f}h</span></div>
            <div class="pos-stat"><span class="pos-stat-label">DTE</span><span class="pos-stat-val">{pos.get('dte', 0):.0f}d</span></div>
            {clv_html}
            <div class="pos-stat"><span class="pos-stat-label">Category</span><span class="pos-stat-val">{category}</span></div>
        </div>
    </div>"""
if not position_cards:
    position_cards = '<div class="empty-state">No open positions · waiting for z≥5 signals (typically 0-2 per cycle)</div>'

# ── Build activity feed (combine cycles + trades) ───────────────
activity_items = []
for t in closed[-15:]:
    pnl = t.get("net_pnl_usd", 0)
    ts = t.get("exit_ts", 0)
    activity_items.append({
        "ts": ts,
        "type": "close",
        "label": f"Closed {'short' if t.get('direction')==-1 else 'long'} {escape(str(t.get('question',''))[:50])} → {fmt_pnl(pnl)} ({t.get('exit_reason','—')})",
    })
for s in signals[-15:]:
    activity_items.append({
        "ts": s.get("ts", 0),
        "type": "open",
        "label": f"Opened {'short' if s.get('direction')==-1 else 'long'} {escape(str(s.get('question',''))[:50])} at z={s.get('z',0):+.2f}",
    })
# Add recent cycles (only "interesting" ones)
for c in recent_cycles[:10]:
    if c.get("signals_at_z5", 0) > 0 or c.get("opened", 0) > 0 or c.get("closed", 0) > 0:
        ts = c.get("ts", 0)
        activity_items.append({
            "ts": ts,
            "type": "cycle",
            "label": f"Scan #{c.get('cycle')}: {c.get('scanned'):,} markets, {c.get('signals_at_z5')} signals, {c.get('opened')} opened, {c.get('closed')} closed",
        })

activity_items.sort(key=lambda x: x["ts"], reverse=True)
activity_items = activity_items[:25]

activity_html = ""
for it in activity_items:
    dt_str = datetime.datetime.fromtimestamp(it["ts"], datetime.timezone.utc).strftime("%m-%d %H:%M")
    icon = {"close": "✗", "open": "▲", "cycle": "↻"}[it["type"]]
    color = {"close": "#a371f7", "open": "#58a6ff", "cycle": "#7d8590"}[it["type"]]
    activity_html += f'<div class="activity-row"><span class="activity-time">{dt_str}</span><span class="activity-icon" style="color:{color}">{icon}</span><span class="activity-text">{it["label"]}</span></div>'
if not activity_html:
    activity_html = '<div class="empty-state-small">No activity yet · waiting for first cycle</div>'

# Recent cycles table
cycle_rows = ""
for c in recent_cycles[:20]:
    ts = c.get("ts", 0)
    dt_str = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%m-%d %H:%M")
    rej = c.get("rejected", {}) or {}
    top_rej = sorted(rej.items(), key=lambda kv: -kv[1])[:2]
    rej_html = " ".join(f'<span class="chip">{k.replace("_", " ")}: {v:,}</span>' for k, v in top_rej if v > 0)
    cycle_rows += f"""
        <tr>
            <td class="mono">{dt_str}</td>
            <td class="mono">#{c.get('cycle', 0)}</td>
            <td class="mono num">{c.get('scanned', 0):,}</td>
            <td class="mono num" style="color:{'#3fb950' if c.get('signals_at_z5', 0) > 0 else '#7d8590'}">{c.get('signals_at_z5', 0)}</td>
            <td class="mono num" style="color:{'#58a6ff' if c.get('opened', 0) > 0 else '#7d8590'}">{c.get('opened', 0)}</td>
            <td class="mono num" style="color:{'#a371f7' if c.get('closed', 0) > 0 else '#7d8590'}">{c.get('closed', 0)}</td>
            <td>{rej_html}</td>
        </tr>"""
if not cycle_rows:
    cycle_rows = '<tr><td colspan="7" class="empty-state-small">No cycles yet</td></tr>'

# Recent trades table
recent_trade_rows = ""
for t in sorted(closed, key=lambda x: x.get("exit_ts", 0), reverse=True)[:30]:
    pnl = t.get("net_pnl_usd", 0)
    exit_dt = datetime.datetime.fromtimestamp(t.get("exit_ts", 0), datetime.timezone.utc).strftime("%m-%d %H:%M")
    spread = t.get("entry_spread", 0) * 100
    stake = (t.get("shares", 0) * t.get("entry_exec_price", 0)) if t.get("direction") == 1 else (t.get("shares", 0) * (1 - t.get("entry_exec_price", 0)))
    clv_text = f"{t.get('clv_value', 0)*100:+.2f}¢" if t.get("clv_value") is not None else "—"
    clv_color = color_for(t.get('clv_value', 0)) if t.get("clv_value") is not None else "#7d8590"
    fees = t.get("total_fees_usd", 0)
    dir_label = "S" if t.get('direction') == -1 else "L"
    dir_color = "#f85149" if t.get('direction') == -1 else "#3fb950"
    recent_trade_rows += f"""
        <tr>
            <td class="mono">{exit_dt}</td>
            <td>{escape(str(t.get('question', ''))[:55])}</td>
            <td class="mono num">{t.get('entry_z', 0):+.1f}σ</td>
            <td class="mono center" style="color:{dir_color}">{dir_label}</td>
            <td class="mono num">{t.get('entry_exec_price', 0):.3f}→{t.get('exit_exec_price', 0):.3f}</td>
            <td class="mono num">{spread:.1f}¢</td>
            <td class="mono num">${stake:.0f}</td>
            <td class="mono num">{t.get('hold_hours', 0):.1f}h</td>
            <td class="mono">{t.get('exit_reason', '—')[:8]}</td>
            <td class="mono num" style="color:{clv_color}">{clv_text}</td>
            <td class="mono num">${fees:.2f}</td>
            <td class="mono num" style="color:{color_for(pnl)};font-weight:600">${pnl:+.2f}</td>
        </tr>"""
if not recent_trade_rows:
    recent_trade_rows = '<tr><td colspan="12" class="empty-state-small">No closed trades yet · z≥5 spikes typically revert in 12-48h</td></tr>'

cat_rows = ""
for cat in sorted(by_cat.keys(), key=lambda k: -len(by_cat[k])):
    arr = by_cat[cat]
    if len(arr) < 1: continue
    cat_pnl = sum(arr)
    cat_wins = sum(1 for x in arr if x > 0)
    cat_winrate = cat_wins / len(arr) * 100
    cat_avg = sum(arr) / len(arr)
    cat_name = cat.replace('_fees', '').replace('_v2', '').replace('_prices', '')
    cat_rows += f"""
        <tr>
            <td>{cat_name}</td>
            <td class="mono num">{len(arr)}</td>
            <td class="mono num">{cat_winrate:.0f}%</td>
            <td class="mono num">${cat_avg:+.2f}</td>
            <td class="mono num" style="color:{color_for(cat_pnl)}">${cat_pnl:+.2f}</td>
        </tr>"""
if not cat_rows:
    cat_rows = '<tr><td colspan="5" class="empty-state-small">No closed trades</td></tr>'

# ── HTML ─────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Polymarket Paper Trader</title>
<meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0a0e14;
    --surface: #11161d;
    --surface-2: #161b22;
    --border: #21262d;
    --border-strong: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --text-mute: #6e7681;
    --green: #3fb950;
    --green-dim: rgba(63, 185, 80, 0.15);
    --red: #f85149;
    --red-dim: rgba(248, 81, 73, 0.15);
    --amber: #d29922;
    --blue: #58a6ff;
    --blue-dim: rgba(88, 166, 255, 0.15);
    --purple: #a371f7;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.4;
  }}
  .mono {{ font-family: "SF Mono", "Menlo", "Monaco", "Roboto Mono", "JetBrains Mono", monospace; font-feature-settings: "tnum" 1; }}
  .num {{ text-align: right; }}
  .center {{ text-align: center; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}

  /* Header */
  header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 28px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
  .logo {{ display: flex; align-items: center; gap: 12px; }}
  .logo-mark {{ width: 36px; height: 36px; background: linear-gradient(135deg, var(--blue) 0%, var(--purple) 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 20px; }}
  .logo-text h1 {{ margin: 0; font-size: 18px; font-weight: 600; letter-spacing: -0.01em; }}
  .logo-text .sub {{ font-size: 11px; color: var(--text-mute); margin-top: 1px; }}
  .live-pill {{ display: flex; align-items: center; gap: 8px; padding: 6px 14px; background: var(--surface); border: 1px solid var(--border); border-radius: 999px; font-size: 12px; }}
  .live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: {liveness_color}; box-shadow: 0 0 0 0 {liveness_color}; animation: pulse 2s infinite; }}
  @keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 {liveness_color}66; }}
    70% {{ box-shadow: 0 0 0 8px transparent; }}
    100% {{ box-shadow: 0 0 0 0 transparent; }}
  }}

  /* Hero */
  .hero {{ display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 1px; background: var(--border); border-radius: 14px; overflow: hidden; margin-bottom: 24px; border: 1px solid var(--border); }}
  .hero-cell {{ background: var(--surface); padding: 22px 24px; }}
  .hero-label {{ font-size: 11px; color: var(--text-mute); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; font-weight: 500; }}
  .hero-value {{ font-size: 36px; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }}
  .hero-sub {{ font-size: 12px; color: var(--text-dim); margin-top: 6px; }}
  .hero-cell-primary {{ background: linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%); }}

  /* Verdict banner */
  .verdict {{ display: flex; align-items: center; gap: 16px; padding: 18px 22px; background: var(--surface); border: 1px solid {verdict_color}33; border-left: 4px solid {verdict_color}; border-radius: 10px; margin-bottom: 24px; }}
  .verdict-icon {{ width: 38px; height: 38px; border-radius: 10px; background: {verdict_color}1a; display: flex; align-items: center; justify-content: center; color: {verdict_color}; font-size: 20px; }}
  .verdict-content {{ flex: 1; }}
  .verdict-label {{ font-weight: 600; color: {verdict_color}; font-size: 15px; }}
  .verdict-sub {{ font-size: 13px; color: var(--text-dim); margin-top: 2px; }}

  /* Cards */
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; }}
  .card-title {{ font-size: 12px; color: var(--text-mute); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 14px; font-weight: 500; display: flex; align-items: center; justify-content: space-between; }}
  .card-title-row {{ display: flex; align-items: center; gap: 8px; }}
  .chip {{ display: inline-block; padding: 2px 8px; background: var(--surface-2); border: 1px solid var(--border-strong); border-radius: 999px; font-size: 10px; color: var(--text-dim); }}

  /* Grid layouts */
  .row-2-1 {{ display: grid; grid-template-columns: 2fr 1fr; gap: 14px; margin-bottom: 14px; }}
  .row-1-1 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }}
  .row-1-1-1 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-bottom: 14px; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-mute); text-transform: uppercase; font-size: 10px; letter-spacing: 0.08em; font-weight: 500; background: transparent; }}
  td {{ color: var(--text); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface-2); }}
  .empty-state {{ padding: 32px; text-align: center; color: var(--text-mute); font-size: 13px; background: var(--surface-2); border-radius: 10px; }}
  .empty-state-small {{ padding: 20px; text-align: center; color: var(--text-mute); font-size: 12px; }}

  /* Position cards */
  .pos-card {{ background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }}
  .pos-card-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 10px; }}
  .pos-question {{ font-size: 13px; color: var(--text); flex: 1; }}
  .pos-dir {{ font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 4px; background: var(--surface); letter-spacing: 0.05em; }}
  .pos-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .pos-stat {{ display: flex; flex-direction: column; }}
  .pos-stat-label {{ font-size: 10px; color: var(--text-mute); text-transform: uppercase; letter-spacing: 0.05em; }}
  .pos-stat-val {{ font-size: 13px; font-weight: 500; font-family: "SF Mono", "Menlo", monospace; margin-top: 2px; }}

  /* Activity feed */
  .activity-row {{ display: grid; grid-template-columns: 80px 24px 1fr; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 12.5px; }}
  .activity-row:last-child {{ border-bottom: none; }}
  .activity-time {{ color: var(--text-mute); font-family: "SF Mono", "Menlo", monospace; font-size: 11px; }}
  .activity-icon {{ text-align: center; font-weight: bold; }}
  .activity-text {{ color: var(--text); }}

  /* Stat detail bar */
  .stat-bar {{ display: flex; gap: 18px; font-size: 12px; color: var(--text-dim); margin-top: 8px; }}
  .stat-bar strong {{ color: var(--text); }}

  /* Chart heights */
  .chart-cell {{ position: relative; height: 240px; }}
  .chart-cell-large {{ position: relative; height: 320px; }}
  .chart-cell-small {{ position: relative; height: 180px; }}

  /* Footer */
  footer {{ margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-mute); display: flex; justify-content: space-between; }}

  /* Sparkline */
  .sparkline {{ height: 28px; margin-top: 8px; }}

  /* Insights strip */
  .insights-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 14px; }}
  .insight-card {{ display: flex; gap: 12px; align-items: center; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; transition: border-color 0.2s, transform 0.2s; }}
  .insight-card:hover {{ border-color: var(--border-strong); transform: translateY(-1px); }}
  .insight-icon {{ width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 16px; flex-shrink: 0; }}
  .insight-body {{ flex: 1; min-width: 0; }}
  .insight-label {{ font-size: 10px; color: var(--text-mute); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }}
  .insight-val {{ font-family: "SF Mono", "Menlo", monospace; font-size: 19px; font-weight: 600; margin-top: 2px; letter-spacing: -0.01em; }}
  .insight-suffix {{ font-size: 13px; color: var(--text-dim); font-weight: 400; }}
  .insight-sub {{ font-size: 11px; color: var(--text-mute); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

  /* Calendar */
  .calendar-grid {{ display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; }}
  .cal-cell {{ aspect-ratio: 1.4; border-radius: 6px; padding: 6px; display: flex; flex-direction: column; justify-content: space-between; font-family: "SF Mono", monospace; transition: transform 0.15s; cursor: default; }}
  .cal-cell:hover {{ transform: scale(1.05); z-index: 5; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
  .cal-day {{ font-size: 10px; opacity: 0.7; }}
  .cal-pnl {{ font-size: 11px; font-weight: 600; }}
  .legend-row {{ display: flex; gap: 16px; font-size: 11px; color: var(--text-mute); }}
  .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 4px; vertical-align: middle; }}

  /* Smooth transitions */
  .card, .hero-cell, .pos-card {{ transition: border-color 0.2s ease; }}
  .card:hover {{ border-color: var(--border-strong); }}

  /* Sparkline (inside hero cell) */
  .sparkline {{ width: 100%; height: 30px; margin-top: 12px; opacity: 0.85; }}

  /* Trade tape */
  .tape-row {{ display: flex; gap: 4px; flex-wrap: wrap; padding: 4px 0; }}
  .tape-dot {{ width: 22px; height: 22px; border-radius: 5px; cursor: default; transition: transform 0.15s; }}
  .tape-dot:hover {{ transform: scale(1.3); z-index: 5; box-shadow: 0 2px 8px rgba(0,0,0,0.5); }}

  /* Time-of-day heatmap */
  .hod-grid {{ display: grid; grid-template-columns: 36px repeat(24, 1fr); gap: 2px; margin-top: 6px; }}
  .hod-corner {{ }}
  .hod-hour-label {{ text-align: center; font-size: 9px; color: var(--text-mute); font-family: "SF Mono", monospace; padding: 2px 0; }}
  .hod-day-label {{ font-size: 10px; color: var(--text-mute); padding-right: 4px; padding-top: 4px; text-align: right; }}
  .hod-cell {{ aspect-ratio: 1; border-radius: 2px; cursor: default; transition: transform 0.15s; }}
  .hod-cell:hover {{ transform: scale(1.5); z-index: 5; box-shadow: 0 2px 8px rgba(0,0,0,0.5); }}

  /* Animated numbers — initial fade in */
  .animated-num {{ animation: fadeUp 0.6s ease-out; }}
  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}

  @media (max-width: 1100px) {{
    .insights-grid {{ grid-template-columns: repeat(3, 1fr); }}
  }}
  @media (max-width: 900px) {{
    .hero {{ grid-template-columns: 1fr 1fr; }}
    .row-2-1, .row-1-1, .row-1-1-1 {{ grid-template-columns: 1fr; }}
    .pos-stats {{ grid-template-columns: repeat(2, 1fr); }}
    .insights-grid {{ grid-template-columns: 1fr 1fr; }}
    .calendar-grid {{ grid-template-columns: repeat(7, 1fr); }}
  }}
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="logo">
      <div class="logo-mark">P</div>
      <div class="logo-text">
        <h1>Polymarket Paper Trader</h1>
        <div class="sub">{now_str} · Cycle #{cycles_run} · Started {started_at[:10] if started_at != '—' else '—'}</div>
      </div>
    </div>
    <div class="live-pill">
      <span class="live-dot"></span>
      <span style="color:{liveness_color};font-weight:600">{liveness}</span>
      <span style="color:var(--text-mute)">·</span>
      <span style="color:var(--text-dim)">{f'{mins_since_last:.0f}m ago' if mins_since_last is not None else 'no data'}</span>
    </div>
  </header>

  <!-- VERDICT -->
  <div class="verdict">
    <div class="verdict-icon">{('✓' if 'CONFIRMED' in verdict_label else '⚠' if 'PENDING' in verdict_label or 'VARIANCE' in verdict_label or 'INSUFFICIENT' in verdict_label or 'LUCKY' in verdict_label else '✗')}</div>
    <div class="verdict-content">
      <div class="verdict-label">{verdict_label}</div>
      <div class="verdict-sub">{verdict_sub}</div>
    </div>
  </div>

  <!-- HERO STRIP -->
  <div class="hero">
    <div class="hero-cell hero-cell-primary">
      <div class="hero-label">Net PnL</div>
      <div class="hero-value mono animated-num" data-target="{total_pnl}" data-prefix="$" data-prec="2" style="color:{color_for(total_pnl)}">${total_pnl:+.2f}</div>
      <div class="hero-sub">{n_closed} trades · ${total_fees:.2f} fees paid · ${'+' if total_pnl-total_fees > 0 else ''}${total_pnl + total_fees:.2f} before fees</div>
      <canvas class="sparkline" id="spark_equity"></canvas>
    </div>
    <div class="hero-cell">
      <div class="hero-label">Avg CLV ★</div>
      <div class="hero-value mono" style="color:{color_for(avg_clv)}">{avg_clv*100:+.2f}¢</div>
      <div class="hero-sub">{n_with_clv} measured · {clv_winrate:.0f}% positive · target +{EXP_CLV:.1f}¢</div>
      <canvas class="sparkline" id="spark_clv"></canvas>
    </div>
    <div class="hero-cell">
      <div class="hero-label">Win Rate</div>
      <div class="hero-value mono">{win_rate:.1f}<span style="font-size:18px;color:var(--text-dim)">%</span></div>
      <div class="hero-sub">{wins}/{n_closed} · expected ~{EXP_WIN_RATE}%</div>
      <canvas class="sparkline" id="spark_winrate"></canvas>
    </div>
    <div class="hero-cell">
      <div class="hero-label">Open Positions</div>
      <div class="hero-value mono">{n_open}<span style="font-size:18px;color:var(--text-dim)">/10</span></div>
      <div class="hero-sub">${total_deployed:.0f} deployed of ${BANKROLL_USD}</div>
      <canvas class="sparkline" id="spark_pos"></canvas>
    </div>
  </div>

  <!-- TRADE TAPE -->
  <div class="card" style="margin-bottom:14px">
    <div class="card-title">
      <div class="card-title-row">📼 Trade tape — last 30</div>
      <span style="color:var(--text-mute);font-size:11px">hover for detail</span>
    </div>
    <div class="tape-row">{tape_html}</div>
  </div>

  <!-- INSIGHTS STRIP -->
  <div class="insights-grid">
    <div class="insight-card">
      <div class="insight-icon" style="background:{'#3fb950' if current_streak_type == 'win' else '#f85149' if current_streak_type == 'loss' else '#7d8590'}33;color:{'#3fb950' if current_streak_type == 'win' else '#f85149' if current_streak_type == 'loss' else '#7d8590'}">{'W' if current_streak_type == 'win' else 'L' if current_streak_type == 'loss' else '—'}</div>
      <div class="insight-body">
        <div class="insight-label">Current streak</div>
        <div class="insight-val">{current_streak if current_streak_type != 'neutral' else 0}<span class="insight-suffix"> {('wins' if current_streak_type == 'win' else 'losses' if current_streak_type == 'loss' else 'trades')}</span></div>
      </div>
    </div>
    <div class="insight-card">
      <div class="insight-icon" style="background:#58a6ff33;color:#58a6ff">★</div>
      <div class="insight-body">
        <div class="insight-label">Best single trade</div>
        <div class="insight-val" style="color:#3fb950">${best_pnl:+.2f}</div>
        <div class="insight-sub">{best_q}</div>
      </div>
    </div>
    <div class="insight-card">
      <div class="insight-icon" style="background:#f8514933;color:#f85149">▼</div>
      <div class="insight-body">
        <div class="insight-label">Worst single trade</div>
        <div class="insight-val" style="color:#f85149">${worst_pnl:+.2f}</div>
        <div class="insight-sub">{worst_q}</div>
      </div>
    </div>
    <div class="insight-card">
      <div class="insight-icon" style="background:#d2992233;color:#d29922">⏱</div>
      <div class="insight-body">
        <div class="insight-label">Next signal expected</div>
        <div class="insight-val">{next_signal_est}</div>
        <div class="insight-sub">{(f'{total_signals_seen} signals seen lifetime') if total_signals_seen else 'no historical data'}</div>
      </div>
    </div>
    <div class="insight-card">
      <div class="insight-icon" style="background:#a371f733;color:#a371f7">⚖</div>
      <div class="insight-body">
        <div class="insight-label">Capital deployed</div>
        <div class="insight-val">{util_pct:.0f}<span class="insight-suffix">%</span></div>
        <div class="insight-sub">${total_deployed:.0f} of ${BANKROLL_USD} bankroll</div>
      </div>
    </div>
  </div>

  <!-- CALENDAR + WATERFALL -->
  <div class="row-2-1">
    <div class="card">
      <div class="card-title">
        <div class="card-title-row">📅 Daily PnL — last 21 days</div>
        <span style="color:var(--text-mute);font-size:11px">brightness = magnitude</span>
      </div>
      <div class="calendar-grid">
        {cal_cells_html}
      </div>
      <div class="legend-row" style="margin-top:10px">
        <div><span class="legend-dot" style="background:rgba(248,81,73,0.7)"></span>Loss</div>
        <div><span class="legend-dot" style="background:#161b22;border:1px solid var(--border)"></span>No trades</div>
        <div><span class="legend-dot" style="background:rgba(63,185,80,0.7)"></span>Profit</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">💵 PnL waterfall</div>
      <div class="chart-cell"><canvas id="waterfall"></canvas></div>
      <div class="stat-bar" style="font-size:11px">
        <div>Gross <strong style="color:{color_for(total_gross)}">${total_gross:+.2f}</strong></div>
        <div>− Fees <strong style="color:var(--red)">${total_fees_only:.2f}</strong></div>
        <div>− Gas <strong style="color:var(--red)">${total_gas:.2f}</strong></div>
        <div>= Net <strong style="color:{color_for(total_net)}">${total_net:+.2f}</strong></div>
      </div>
    </div>
  </div>

  <!-- MAIN: EQUITY + ACTIVITY FEED -->
  <div class="row-2-1">
    <div class="card">
      <div class="card-title">
        <div class="card-title-row">📈 Equity curve <span class="chip">live vs backtest</span></div>
        <span style="color:var(--text-mute);font-size:11px">${total_pnl:+.0f} cumulative · {n_closed} trades</span>
      </div>
      <div class="chart-cell-large"><canvas id="equity"></canvas></div>
      <div class="stat-bar">
        <div>Max drawdown <strong style="color:{color_for(max_dd)}">${max_dd:+.2f}</strong></div>
        <div>Avg trade <strong style="color:{color_for(avg_trade)}">${avg_trade:+.2f}</strong></div>
        <div>Best <strong style="color:var(--green)">${max(pnls, default=0):+.2f}</strong></div>
        <div>Worst <strong style="color:var(--red)">${min(pnls, default=0):+.2f}</strong></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">⚡ Activity feed</div>
      <div style="max-height:380px;overflow-y:auto">
        {activity_html}
      </div>
    </div>
  </div>

  <!-- STRATEGY ANALYTICS ROW -->
  <div class="row-2-1">
    <div class="card">
      <div class="card-title">
        <div class="card-title-row">🕐 Time-of-day performance heatmap</div>
        <span style="color:var(--text-mute);font-size:11px">UTC · all days × all hours</span>
      </div>
      {hod_html}
      <div class="legend-row" style="margin-top:8px">
        <div><span class="legend-dot" style="background:rgba(248,81,73,0.7)"></span>Loss hours</div>
        <div><span class="legend-dot" style="background:#161b22;border:1px solid var(--border)"></span>Unused</div>
        <div><span class="legend-dot" style="background:rgba(63,185,80,0.7)"></span>Profit hours</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">
        <div class="card-title-row">🥧 Open portfolio</div>
        <span style="color:var(--text-mute);font-size:11px">stake by category</span>
      </div>
      <div class="chart-cell"><canvas id="portfolio"></canvas></div>
    </div>
  </div>

  <!-- STREAK + OPEN POSITIONS -->
  <div class="row-1-1">
    <div class="card">
      <div class="card-title">
        <div class="card-title-row">📊 Streak history</div>
        <span style="color:var(--text-mute);font-size:11px">consecutive W/L runs</span>
      </div>
      <div class="chart-cell-small"><canvas id="streaks"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">
        <div class="card-title-row">📈 PnL distribution per trade</div>
      </div>
      <div class="chart-cell-small"><canvas id="pnlhist"></canvas></div>
    </div>
  </div>

  <!-- OPEN POSITIONS -->
  <div class="card" style="margin-bottom:14px">
    <div class="card-title">
      <div class="card-title-row">🎯 Open positions <span class="chip">{n_open}/10</span></div>
    </div>
    {position_cards}
  </div>

  <!-- BOT HEALTH + LIFETIME STATS -->
  <div class="row-2-1">
    <div class="card">
      <div class="card-title">
        <div class="card-title-row">🫀 Recent cycles (heartbeat)</div>
        <span style="color:var(--text-mute);font-size:11px">scan every ~15min</span>
      </div>
      <div style="max-height:340px;overflow-y:auto">
        <table>
          <thead>
            <tr><th>Time</th><th>Cycle</th><th class="num">Markets</th><th class="num">Signals</th><th class="num">Opened</th><th class="num">Closed</th><th>Filters</th></tr>
          </thead>
          <tbody>{cycle_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title">📊 Lifetime totals</div>
      <div style="display:grid;grid-template-columns:1fr;gap:12px;font-size:13px">
        <div><div style="color:var(--text-mute);font-size:11px;text-transform:uppercase;letter-spacing:0.05em">Markets scanned</div><div class="mono" style="font-size:22px;font-weight:600">{total_markets_scanned:,}</div></div>
        <div><div style="color:var(--text-mute);font-size:11px;text-transform:uppercase;letter-spacing:0.05em">z≥5 signals seen</div><div class="mono" style="font-size:22px;font-weight:600">{total_signals_seen}</div></div>
        <div><div style="color:var(--text-mute);font-size:11px;text-transform:uppercase;letter-spacing:0.05em">Conversion (signals → entries)</div><div class="mono" style="font-size:22px;font-weight:600">{(100*sum(c.get('opened',0) for c in cycles)/total_signals_seen) if total_signals_seen else 0:.1f}%</div></div>
        <div><div style="color:var(--text-mute);font-size:11px;text-transform:uppercase;letter-spacing:0.05em">Filter rejection split</div>
          <div style="margin-top:4px">
            {''.join(f'<span class="chip" style="margin:2px 4px 2px 0">{k.replace("_", " ")}: <strong style="color:var(--text)">{v:,}</strong></span>' for k, v in sorted(lifetime_rejections.items(), key=lambda x: -x[1])[:8] if v > 0)}
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- CHARTS GRID -->
  <div class="row-1-1-1">
    <div class="card">
      <div class="card-title">📉 Drawdown</div>
      <div class="chart-cell"><canvas id="drawdown"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">🎯 Rolling 20-trade win rate</div>
      <div class="chart-cell"><canvas id="rollwin"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">🔮 Forecast cone (next 100 trades)</div>
      <div class="chart-cell"><canvas id="forecast"></canvas></div>
    </div>
  </div>

  <div class="card" style="margin-bottom:14px">
    <div class="card-title">🏷 Performance by market category</div>
    <table>
      <thead>
        <tr><th>Category</th><th class="num">n</th><th class="num">Win%</th><th class="num">Avg</th><th class="num">Total</th></tr>
      </thead>
      <tbody>{cat_rows}</tbody>
    </table>
  </div>

  <!-- RECENT TRADES -->
  <div class="card" style="margin-top:14px">
    <div class="card-title">📋 Recent closed trades</div>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Exit</th><th>Market</th><th class="num">z</th><th class="center">Dir</th>
            <th class="num">Px in→out</th><th class="num">Spread</th><th class="num">Stake</th>
            <th class="num">Held</th><th>Reason</th><th class="num">CLV</th><th class="num">Fees</th><th class="num">Net</th>
          </tr>
        </thead>
        <tbody>{recent_trade_rows}</tbody>
      </table>
    </div>
  </div>

  <footer>
    <div>Auto-refresh 60s · v1 strategy (z≥5, both directions, max 48h hold)</div>
    <div>{n_closed} closed · {n_open} open · {cycles_run} cycles</div>
  </footer>

</div>

<script>
const DARK = {{
  text: '#e6edf3', textDim: '#8b949e', textMute: '#6e7681',
  bg: '#11161d', surface: '#161b22', border: '#30363d',
  green: '#3fb950', red: '#f85149', blue: '#58a6ff', purple: '#a371f7', amber: '#d29922'
}};

Chart.defaults.color = DARK.textDim;
Chart.defaults.borderColor = DARK.border;
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, Inter, system-ui';
Chart.defaults.scale.grid.color = DARK.border + '40';
Chart.defaults.scale.grid.drawBorder = false;
Chart.defaults.plugins.legend.display = false;

const COMMON = {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
        legend: {{ display: false }},
        tooltip: {{ mode: 'index', intersect: false, backgroundColor: '#0a0e14', borderColor: DARK.border, borderWidth: 1, titleColor: DARK.text, bodyColor: DARK.text, padding: 10 }}
    }},
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
                {{ label: 'Expected', data: baseline_y, borderColor: DARK.textMute, borderDash: [4, 4], pointRadius: 0, fill: false, borderWidth: 1.5 }},
                {{ label: 'Break-even', data: equity_y.map(() => 0), borderColor: DARK.red + '66', borderDash: [2, 4], pointRadius: 0, fill: false, borderWidth: 1 }},
                {{ label: 'Live', data: equity_y, borderColor: DARK.blue, backgroundColor: DARK.blue + '22', fill: true, tension: 0.2, pointRadius: 3, pointBackgroundColor: DARK.blue, borderWidth: 2 }},
            ]
        }},
        options: {{ ...COMMON, scales: {{ x: {{ ticks: {{ maxTicksLimit: 8 }} }}, y: {{ ticks: {{ callback: (v) => '$' + v }} }} }} }}
    }});
}} else {{
    document.getElementById('equity').parentElement.innerHTML = '<div class="empty-state">No closed trades yet · waiting for first z≥5 signal to revert</div>';
}}

// Drawdown
const drawdown_y = {json.dumps(drawdown_y)};
if (drawdown_y.length > 0) {{
    new Chart(document.getElementById('drawdown'), {{
        type: 'line',
        data: {{ labels: equity_x, datasets: [{{ data: drawdown_y, borderColor: DARK.red, backgroundColor: DARK.red + '22', fill: 'origin', pointRadius: 0, borderWidth: 2 }}] }},
        options: {{ ...COMMON, scales: {{ x: {{ ticks: {{ maxTicksLimit: 6 }} }}, y: {{ max: 0, ticks: {{ callback: (v) => '$' + v }} }} }} }}
    }});
}} else {{
    document.getElementById('drawdown').parentElement.innerHTML = '<div class="empty-state-small">no data</div>';
}}

// Rolling win rate
const rw_x = {json.dumps(rolling_winrate_x)};
const rw_y = {json.dumps(rolling_winrate_y)};
if (rw_y.length > 0) {{
    new Chart(document.getElementById('rollwin'), {{
        type: 'line',
        data: {{ labels: rw_x, datasets: [
            {{ data: rw_y.map(() => {EXP_WIN_RATE}), borderColor: DARK.textMute, borderDash: [4, 4], pointRadius: 0, fill: false, borderWidth: 1 }},
            {{ data: rw_y, borderColor: DARK.green, backgroundColor: DARK.green + '22', fill: true, pointRadius: 2, borderWidth: 2 }},
        ] }},
        options: {{ ...COMMON, scales: {{ y: {{ min: 0, max: 100, ticks: {{ callback: (v) => v + '%' }} }} }} }}
    }});
}} else {{
    document.getElementById('rollwin').parentElement.innerHTML = '<div class="empty-state-small">need 20+ trades</div>';
}}

// Forecast cone
const proj_p5 = {json.dumps(projections_p5)};
const proj_p50 = {json.dumps(projections_p50)};
const proj_p95 = {json.dumps(projections_p95)};
const n_closed = {n_closed};
const labels_forecast = [];
for (let i = 1; i <= n_closed + proj_p50.length; i++) labels_forecast.push(i);
const live_p = equity_y.concat(proj_p50.map(() => null));
const p5_p = equity_y.map(() => null).concat(proj_p5);
const p50_p = equity_y.map(() => null).concat(proj_p50);
const p95_p = equity_y.map(() => null).concat(proj_p95);

new Chart(document.getElementById('forecast'), {{
    type: 'line',
    data: {{ labels: labels_forecast, datasets: [
        {{ data: p95_p, borderColor: DARK.green + '66', backgroundColor: DARK.green + '15', fill: '+1', pointRadius: 0, borderWidth: 1 }},
        {{ data: p5_p, borderColor: DARK.green + '66', fill: false, pointRadius: 0, borderWidth: 1 }},
        {{ data: p50_p, borderColor: DARK.green, pointRadius: 0, fill: false, borderWidth: 2 }},
        {{ data: live_p, borderColor: DARK.blue, backgroundColor: DARK.blue + '22', pointRadius: 1.5, fill: false, borderWidth: 2 }},
    ] }},
    options: {{ ...COMMON, scales: {{ y: {{ ticks: {{ callback: (v) => '$' + v }} }} }} }}
}});

// Waterfall: gross → -fees → -gas → net
const wGross = {total_gross};
const wFees = {total_fees_only};
const wGas = {total_gas};
const wNet = {total_net};
if (wGross !== 0 || wFees !== 0 || wGas !== 0) {{
    new Chart(document.getElementById('waterfall'), {{
        type: 'bar',
        data: {{
            labels: ['Gross', '− Fees', '− Gas', 'Net'],
            datasets: [{{
                data: [wGross, -wFees, -wGas, wNet],
                backgroundColor: [
                    wGross >= 0 ? DARK.green + 'cc' : DARK.red + 'cc',
                    DARK.red + 'aa',
                    DARK.red + 'aa',
                    wNet >= 0 ? DARK.green : DARK.red
                ],
                borderWidth: 0,
                borderRadius: 4,
            }}]
        }},
        options: {{ ...COMMON, scales: {{ y: {{ ticks: {{ callback: (v) => '$' + v }} }} }} }}
    }});
}} else {{
    document.getElementById('waterfall').parentElement.innerHTML += '<div class="empty-state-small">No trades yet</div>';
}}

// ── Sparklines (inside hero cells) ──
const sparkOpts = {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }},
    scales: {{ x: {{ display: false }}, y: {{ display: false }} }},
    elements: {{ point: {{ radius: 0 }}, line: {{ borderWidth: 1.5, tension: 0.4 }} }}
}};
function mkSpark(id, data, color) {{
    if (!data || data.length < 2) return;
    new Chart(document.getElementById(id), {{
        type: 'line',
        data: {{ labels: data.map((_, i) => i), datasets: [{{ data: data, borderColor: color, backgroundColor: color + '20', fill: true }}] }},
        options: sparkOpts
    }});
}}
mkSpark('spark_equity', {json.dumps(spark_equity)}, '{('#3fb950' if total_pnl >= 0 else '#f85149')}');
mkSpark('spark_clv', {json.dumps(spark_clv)}, '{('#3fb950' if avg_clv >= 0 else '#f85149')}');
mkSpark('spark_winrate', {json.dumps(spark_winrate)}, '#58a6ff');
mkSpark('spark_pos', {json.dumps(spark_positions)}, '#a371f7');

// ── Portfolio donut ──
const portLabels = {json.dumps(portfolio_labels)};
const portValues = {json.dumps(portfolio_values)};
if (portValues.length > 0) {{
    new Chart(document.getElementById('portfolio'), {{
        type: 'doughnut',
        data: {{
            labels: portLabels,
            datasets: [{{
                data: portValues,
                backgroundColor: ['#58a6ff', '#3fb950', '#a371f7', '#d29922', '#f85149', '#79c0ff', '#7ee787', '#ff7b72'],
                borderColor: '#0a0e14', borderWidth: 2
            }}]
        }},
        options: {{
            ...COMMON,
            plugins: {{
                ...COMMON.plugins,
                legend: {{ display: true, position: 'right', labels: {{ color: DARK.text, font: {{ size: 11 }} }} }},
                tooltip: {{ callbacks: {{ label: (ctx) => `${{ctx.label}}: $${{ctx.parsed.toFixed(2)}}` }} }}
            }},
            cutout: '60%'
        }}
    }});
}} else {{
    document.getElementById('portfolio').parentElement.innerHTML = '<div class="empty-state">No open positions</div>';
}}

// ── Streak history ──
const streakData = {json.dumps(streak_history)};
if (streakData.length > 0) {{
    new Chart(document.getElementById('streaks'), {{
        type: 'bar',
        data: {{
            labels: streakData.map((_, i) => i + 1),
            datasets: [{{
                data: streakData,
                backgroundColor: streakData.map(v => v > 0 ? DARK.green + 'cc' : DARK.red + 'cc'),
                borderWidth: 0,
                borderRadius: 3,
            }}]
        }},
        options: {{
            ...COMMON,
            scales: {{ y: {{ ticks: {{ callback: (v) => Math.abs(v) }} }} }},
            plugins: {{ ...COMMON.plugins, tooltip: {{ callbacks: {{ label: (ctx) => (ctx.parsed.y > 0 ? `${{ctx.parsed.y}} wins in a row` : `${{Math.abs(ctx.parsed.y)}} losses in a row`) }} }} }}
        }}
    }});
}} else {{
    document.getElementById('streaks').parentElement.innerHTML += '<div class="empty-state-small">No streaks yet · need closed trades</div>';
}}

// ── Animated counter (count up on load for net PnL hero) ──
document.querySelectorAll('.animated-num').forEach(el => {{
    const target = parseFloat(el.dataset.target);
    if (isNaN(target)) return;
    const prefix = el.dataset.prefix || '';
    const prec = parseInt(el.dataset.prec || '0');
    const duration = 800;
    const start = performance.now();
    const fmt = (v) => (v >= 0 ? '+' : '') + prefix + v.toFixed(prec);
    function tick(t) {{
        const p = Math.min(1, (t - start) / duration);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = fmt(target * eased);
        if (p < 1) requestAnimationFrame(tick);
        else el.textContent = fmt(target);
    }}
    requestAnimationFrame(tick);
}});

// PnL histogram
const pnls = {json.dumps(pnls)};
if (pnls.length > 0) {{
    const bins = 25; const min = Math.min(...pnls), max = Math.max(...pnls);
    const range = max - min || 1; const w = range / bins;
    const buckets = new Array(bins).fill(0); const labels = [];
    for (let i = 0; i < bins; i++) labels.push((min + w * (i + 0.5)).toFixed(1));
    pnls.forEach(p => buckets[Math.min(bins - 1, Math.floor((p - min) / w))]++);
    new Chart(document.getElementById('pnlhist'), {{
        type: 'bar',
        data: {{ labels: labels, datasets: [{{ data: buckets, backgroundColor: labels.map(l => parseFloat(l) >= 0 ? DARK.green + 'cc' : DARK.red + 'cc'), borderWidth: 0 }}] }},
        options: COMMON
    }});
}} else {{
    document.getElementById('pnlhist').parentElement.innerHTML = '<div class="empty-state-small">no trades</div>';
}}
</script>

</body>
</html>
"""

OUT.write_text(html)
print(f"Generated {OUT}")
print(f"  closed: {n_closed}  open: {n_open}  cycles: {len(cycles)}  total_pnl: ${total_pnl:+.2f}")
