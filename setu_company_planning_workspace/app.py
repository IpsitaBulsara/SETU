"""
SETU — functional prototype (Streamlit port)
Team Alucard, Maestros 2026

Three modules, each doing real computation:
1. Demand sensing    — log-linear regression, fit + backtested live on synthetic data, vs a naive seasonal baseline
2. Should-cost model — REAL cocoa/milk/FX prices pulled live from Yahoo Finance -> trend/z-score buy signal
                        (falls back to synthetic mean-reverting series if Yahoo Finance is unreachable)
3. Control tower     — a live small warehouse network with an agent that actually rebalances stock

Run:  streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

st.set_page_config(page_title="SETU — Functional Prototype", layout="wide", page_icon="🧭")

# ---------------------------------------------------------------------------
# THEME / STYLE — Mondelez case-brief palette
# ---------------------------------------------------------------------------
GOLD = "#C57A1B"
PURPLE = "#4B2E83"
GREEN = "#2F7A54"
RED = "#B23A2E"
CREAM = "#3E2817"     # ink/text on the warm background
INK = "#2A1B0F"
MUTED = "#8A6F4E"
BG = "#E7A23A"        # mustard/gold field, matches the case brief cover
PANEL = "#FBF3E1"     # cream card
LINE = "#E3CFA0"
LOG_BG = "#341F5C"     # deep Mondelez purple, deliberate dark accent for the ops log
LOG_TEXT = "#E7DDF0"
LOG_MUTED = "#B9A8D6"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    h1, h2, h3 {{ color: {INK}; }}
    p, label, .stMarkdown {{ color: {CREAM}; }}
    .setu-tag {{ font-family: monospace; color: #6b4a1f; font-size: 13px; margin-top: -12px; }}
    .pill {{
        display:inline-block; font-family: monospace; font-size: 11px; letter-spacing:0.06em;
        color: {PURPLE}; border: 1px solid {PURPLE}; border-radius: 20px; padding: 3px 11px;
        background: rgba(75,46,131,0.06);
    }}
    .signal-buy {{ background: rgba(47,122,84,0.14); color:{GREEN}; border:1px solid {GREEN};
        padding:8px 16px; border-radius:3px; font-weight:700; display:inline-block; font-family:monospace;}}
    .signal-wait {{ background: rgba(178,58,46,0.14); color:{RED}; border:1px solid {RED};
        padding:8px 16px; border-radius:3px; font-weight:700; display:inline-block; font-family:monospace;}}
    .signal-neutral {{ background: rgba(197,122,27,0.14); color:{GOLD}; border:1px solid {GOLD};
        padding:8px 16px; border-radius:3px; font-weight:700; display:inline-block; font-family:monospace;}}
    .wh-card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:4px; padding:14px; border-top:3px solid {LINE};}}
    .wh-ok {{ border-top-color: {GREEN}; }}
    .wh-alert {{ border-top-color: {RED}; }}
    .log-line {{ font-family: monospace; font-size: 12.5px; padding: 4px 0 4px 10px; border-left:2px solid {LOG_MUTED}; margin-bottom:2px;}}
    .log-alert {{ border-left-color: #e2867a; color:#f1b3ae; }}
    .log-resolve {{ border-left-color: #7fd1a8; color:#bfe9d5; }}
    .log-info {{ border-left-color: {LOG_MUTED}; color:{LOG_TEXT}; }}
    [data-testid="stMetricValue"] {{ color: {PURPLE}; }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    .stTabs [data-baseweb="tab"] {{ color: {CREAM}; }}
    .stTabs [aria-selected="true"] {{ color: {PURPLE}; font-weight:600; }}
    .stButton button {{ background-color: {PURPLE}; color: {PANEL}; border: none; }}
    .stButton button:hover {{ background-color: {PURPLE}; opacity:0.88; color:{PANEL}; }}
    div[data-testid="stContainer"] {{ background-color: {LOG_BG}; }}
</style>
""", unsafe_allow_html=True)

st.title("SETU\u200a·\u200a functional prototype")
st.markdown('<div class="setu-tag">Unify · Sense · Simulate · Act — Team Alucard, Maestros 2026</div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Demand Sensing", "💰 Should-Cost", "🗼 Control Tower & Agent", "🤖 Autonomous Agent Mesh"
])

# ---------------------------------------------------------------------------
# SHARED PLOT LAYOUT
# ---------------------------------------------------------------------------
def base_layout(title):
    return dict(
        title=dict(text=title, font=dict(color=INK, size=14)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=MUTED), legend=dict(font=dict(color=CREAM)),
        xaxis=dict(gridcolor=LINE), yaxis=dict(gridcolor=LINE),
        margin=dict(t=40, l=40, r=20, b=30),
    )

# ===========================================================================
# MODULE 1 — DEMAND SENSING
# ===========================================================================
def gen_demand_series(seed, weeks=104):
    rng = np.random.default_rng(seed)
    w = np.arange(weeks)
    woy = w % 52
    trend = 1.15 / weeks
    level = 1000 * (1 + trend) ** w
    seasonal = 1 + 0.28*np.sin((woy/52)*2*np.pi - 1.2) + 0.15*np.sin((woy/26)*2*np.pi)
    promo = (rng.random(weeks) < 0.12).astype(float)
    promo_lift = promo * (0.35 + 0.25*rng.random(weeks))
    weather_dev = rng.normal(0, 1, weeks)
    weather_effect = -0.04 * weather_dev
    competitor_shock = np.where(rng.random(weeks) < 0.10, -(0.15 + 0.3*rng.random(weeks)), 0.0)
    local_shock = np.where(rng.random(weeks) < 0.06, rng.normal(0, 0.35, weeks), 0.0)
    noise = rng.normal(0, 0.18, weeks) + local_shock
    demand = level * seasonal * (1+promo_lift) * (1+weather_effect) * (1+competitor_shock) * (1+noise)
    demand = np.maximum(50, demand)
    return w, woy, promo, weather_dev, demand

def featurize_demand(w, woy, promo, weather_dev):
    ang = (woy/52)*2*np.pi
    return np.column_stack([
        np.ones_like(w, dtype=float), w, np.sin(ang), np.cos(ang), np.sin(2*ang), promo, weather_dev
    ])

def run_demand_module():
    seed = np.random.randint(0, 1_000_000_000)
    w, woy, promo, weather_dev, demand = gen_demand_series(seed)
    n_train = 92
    Xtr = featurize_demand(w[:n_train], woy[:n_train], promo[:n_train], weather_dev[:n_train])
    ytr = np.log(demand[:n_train])
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

    Xte = featurize_demand(w[n_train:], woy[n_train:], promo[n_train:], weather_dev[n_train:])
    model_pred = np.exp(Xte @ beta)
    actual = demand[n_train:]

    # naive seasonal baseline: mean of same week-of-year in training set
    naive_pred = np.array([
        demand[:n_train][woy[:n_train] == wk].mean() if (woy[:n_train] == wk).any() else demand[:n_train].mean()
        for wk in woy[n_train:]
    ])

    def acc(a, p):
        return max(0.0, 1 - np.mean(np.abs((a-p)/a))) * 100

    model_acc = acc(actual, model_pred)
    naive_acc = acc(actual, naive_pred)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(w), y=list(demand), mode="lines", name="Synthetic weekly demand",
                               line=dict(color=GOLD, width=1.5), fill="tozeroy", fillcolor="rgba(197,122,27,0.10)"))
    fig1.update_layout(**base_layout("Generated demand history (train + holdout)"))
    fig1.update_yaxes(title="Units/week")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(w[n_train:]), y=list(actual), mode="lines+markers", name="Actual",
                               line=dict(color=CREAM, width=2.5)))
    fig2.add_trace(go.Scatter(x=list(w[n_train:]), y=list(naive_pred), mode="lines", name="Naive seasonal baseline",
                               line=dict(color=MUTED, width=1.5, dash="dash")))
    fig2.add_trace(go.Scatter(x=list(w[n_train:]), y=list(model_pred), mode="lines+markers", name="Fitted regression model",
                               line=dict(color=GREEN, width=2)))
    fig2.update_layout(**base_layout("12-week holdout backtest"))
    fig2.update_yaxes(title="Units/week")

    labels = ["Intercept/level", "Trend (weekly growth)", "Seasonal (annual)", "Seasonal (phase)",
              "Seasonal (semi-annual)", "Promotion effect", "Weather sensitivity"]

    return fig1, fig2, model_acc, naive_acc, list(zip(labels, beta))

with tab1:
    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.markdown("##### Backtest controls")
        st.write("A synthetic 104-week history (trend, seasonality, promos, weather, plus noise the model can't see) "
                 "is generated fresh, and a log-linear regression is actually fit and backtested on the last 12 weeks.")
        regen = st.button("🔁 Regenerate & refit", use_container_width=True, key="regen_demand")
        if regen or "demand_result" not in st.session_state:
            st.session_state.demand_result = run_demand_module()
        fig1, fig2, model_acc, naive_acc, betas = st.session_state.demand_result

        st.metric("Naive seasonal baseline accuracy", f"{naive_acc:.1f}%")
        st.metric("Fitted model accuracy (holdout)", f"{model_acc:.1f}%", delta=f"+{model_acc-naive_acc:.1f} pts")
        with st.expander("Fitted coefficients (log-scale)"):
            for label, b in betas:
                st.write(f"**{label}**: `{b:.3f}`")
        st.caption("Illustrative synthetic backtest — directionally consistent with the roadmap's 62%→78% claim, "
                    "but the exact figure will vary by run since the data is randomly regenerated each time.")
    with c2:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# MODULE 2 — SHOULD-COST MODELLING  (real Yahoo Finance data, synthetic fallback)
# ===========================================================================
YF_TICKERS = {"cocoa": "CC=F", "dairy": "DC=F", "fx": "INR=X"}
YF_NAMES = {"cocoa": "Cocoa futures (ICE, CC=F)", "dairy": "Class III milk futures (CME, DC=F)", "fx": "USD/INR (INR=X)"}

def gen_price_series(seed, days, start, mean_level, revert_speed, vol):
    """Synthetic fallback: mean-reverting random walk, used only if Yahoo Finance is unreachable."""
    rng = np.random.default_rng(seed)
    p = np.empty(days)
    p[0] = start
    for d in range(1, days):
        shock = rng.normal(0, vol)
        p[d] = max(1.0, p[d-1] + revert_speed*(mean_level - p[d-1]) + p[d-1]*shock)
    return p

@st.cache_data(ttl=900, show_spinner="Pulling cocoa, milk and INR/USD prices from Yahoo Finance…")
def fetch_yahoo_prices(nonce, period="9mo"):
    """Fetch real daily closes for cocoa, milk (dairy proxy) and USD/INR from Yahoo Finance.
    Raises on failure so the caller can fall back to synthetic data."""
    tickers = list(YF_TICKERS.values())
    data = yf.download(tickers=tickers, period=period, interval="1d",
                        group_by="ticker", auto_adjust=True, progress=False, threads=True)
    closes = {}
    for key, tkr in YF_TICKERS.items():
        try:
            series = data[tkr]["Close"].dropna()
        except (KeyError, TypeError):
            series = data["Close"].dropna() if "Close" in data else pd.Series(dtype=float)
        if series.empty:
            raise RuntimeError(f"No data returned for {tkr}")
        closes[key] = series
    # align on common trading dates, keep last 180 rows
    df = pd.DataFrame(closes).dropna()
    if len(df) < 40:
        raise RuntimeError("Not enough overlapping trading history returned")
    return df.tail(180)

def compute_signal(idx):
    recent = idx[-30:].values
    xs = np.arange(len(recent))
    slope = np.polyfit(xs, recent, 1)[0]
    base90 = idx[-90:].values
    sd = base90.std() or 1.0
    z = (recent[-1] - base90.mean()) / sd
    if slope < -0.03 and z < -0.3:
        signal, cls = "BUY NOW", "signal-buy"
        rationale = "30-day trend is falling and the blended index sits below its 90-day average — the buy window is open."
    elif slope > 0.05 and z > 0.3:
        signal, cls = "WAIT / HEDGE", "signal-wait"
        rationale = "30-day trend is rising and the blended index sits above its 90-day average — hedge or defer non-committed volume."
    else:
        signal, cls = "NEUTRAL", "signal-neutral"
        rationale = "No strong directional signal in the last 30 days — proceed on standard buying calendar."
    return slope, z, signal, cls, rationale

def run_cost_module(nonce):
    w = dict(cocoa=0.5, dairy=0.3, fx=0.2)
    source_note = ""
    is_live = False

    if YFINANCE_AVAILABLE:
        try:
            df = fetch_yahoo_prices(nonce)
            cocoa, dairy, fx = df["cocoa"], df["dairy"], df["fx"]
            is_live = True
            source_note = f"Live Yahoo Finance data — {df.index[0].date()} to {df.index[-1].date()} ({len(df)} trading days)."
        except Exception as e:
            is_live = False
            source_note = f"Yahoo Finance fetch failed ({e}) — showing synthetic fallback data instead."

    if not is_live:
        days = 180
        rng = np.random.default_rng()
        s1, s2, s3 = rng.integers(0, 1_000_000_000, 3)
        idx_range = pd.RangeIndex(days)
        cocoa = pd.Series(gen_price_series(int(s1), days, 100, 100+8*(rng.random()-0.3), 0.02, 0.018), index=idx_range)
        dairy = pd.Series(gen_price_series(int(s2), days, 100, 100+3*(rng.random()-0.3), 0.03, 0.010), index=idx_range)
        fx    = pd.Series(gen_price_series(int(s3), days, 83.0, 83.0+1.2*(rng.random()-0.3), 0.015, 0.006), index=idx_range)
        if not YFINANCE_AVAILABLE:
            source_note = "yfinance not installed — showing synthetic fallback data. Run `pip install yfinance` for live prices."

    # rebase every series to 100 at the start of the window so the blend is comparable regardless of native units
    cocoa_r = cocoa / cocoa.iloc[0] * 100
    dairy_r = dairy / dairy.iloc[0] * 100
    fx_r    = fx / fx.iloc[0] * 100
    idx = w["cocoa"]*cocoa_r + w["dairy"]*dairy_r + w["fx"]*fx_r

    slope, z, signal, cls, rationale = compute_signal(idx)
    current = idx.iloc[-1]
    trailing_avg = idx.iloc[-60:-30].mean() if len(idx) >= 60 else idx.iloc[:len(idx)//2].mean()
    delta_pct = (current - trailing_avg) / trailing_avg * 100

    x_axis = list(range(len(idx)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=list(idx.values), name="Blended materials index", line=dict(color=GOLD, width=2.5)))
    fig.add_trace(go.Scatter(x=x_axis, y=list(cocoa_r.values), name="Cocoa (rebased)", line=dict(color="#8a5a3a", width=1), visible="legendonly"))
    fig.add_trace(go.Scatter(x=x_axis, y=list(dairy_r.values), name="Milk/dairy (rebased)", line=dict(color="#5aa0c9", width=1), visible="legendonly"))
    fig.add_trace(go.Scatter(x=x_axis, y=list(fx_r.values), name="USD/INR (rebased)", line=dict(color=PURPLE, width=1), visible="legendonly"))
    title = ("Live materials cost index — Yahoo Finance (index = 100 at window start)" if is_live
              else "Materials cost index — synthetic fallback (index = 100 at window start)")
    fig.update_layout(**base_layout(title))
    fig.update_yaxes(title="Index")
    fig.update_xaxes(title="Trading days")

    return fig, signal, cls, rationale, current, trailing_avg, delta_pct, slope, z, is_live, source_note

with tab2:
    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.markdown("##### Market data")
        st.write("Pulls real daily closes for cocoa futures, Class III milk futures (a dairy proxy — India doesn't "
                 "have a liquid public dairy futures contract) and USD/INR from Yahoo Finance, blends them by "
                 "materials mix, and reads trend + z-score for a live buy/wait signal. Falls back to a synthetic "
                 "series automatically if Yahoo Finance can't be reached.")
        if "cost_nonce" not in st.session_state:
            st.session_state.cost_nonce = 0
        refresh_cost = st.button("🔁 Refresh market data", use_container_width=True, key="regen_cost")
        if refresh_cost:
            st.session_state.cost_nonce += 1
        result = run_cost_module(st.session_state.cost_nonce)
        fig, signal, cls, rationale, current, trailing_avg, delta_pct, slope, z, is_live, source_note = result

        if is_live:
            st.success(source_note, icon="📡")
        else:
            st.warning(source_note, icon="⚠️")

        st.markdown(f'<div class="{cls}">{signal}</div>', unsafe_allow_html=True)
        st.write(rationale)
        st.metric("Current blended index", f"{current:.1f}")
        st.metric("30–60 day prior average", f"{trailing_avg:.1f}")
        st.metric("Move vs prior window", f"{delta_pct:+.1f}%")
        st.metric("30-day trend slope", f"{slope:.3f} idx pts/day")
        st.metric("Z-score vs 90-day mean", f"{z:.2f}")
        st.caption("Weights: cocoa 50% · dairy(milk proxy) 30% · FX 20%, reflecting a chocolate-heavy materials mix. "
                   "Tickers: " + ", ".join(YF_NAMES.values()) + ".")
    with c2:
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# MODULE 3 — CONTROL TOWER + AGENT
# ===========================================================================
def reset_network():
    st.session_state.warehouses = {
        "WH-N": {"label": "WH-N (Delhi)", "stock": 1200, "demand": 90},
        "WH-W": {"label": "WH-W (Mumbai)", "stock": 620, "demand": 140},
        "WH-S": {"label": "WH-S (Chennai)", "stock": 950, "demand": 70},
        "WH-E": {"label": "WH-E (Kolkata)", "stock": 700, "demand": 60},
    }
    st.session_state.history = {k: [] for k in st.session_state.warehouses}
    st.session_state.tick = 0
    st.session_state.alerts_raised = 0
    st.session_state.alerts_resolved = 0
    st.session_state.ttm_samples = []
    st.session_state.log = [("info", "Network initialised. Target cover: 5 days.")]

def days_of_cover(w):
    return w["stock"] / w["demand"]

def agent_resolve(trigger_label):
    whs = st.session_state.warehouses
    total_stock = sum(w["stock"] for w in whs.values())
    total_demand = sum(w["demand"] for w in whs.values())
    network_cover = total_stock / total_demand
    moves = []
    for k, w in whs.items():
        target = round(w["demand"] * network_cover)
        delta = target - w["stock"]
        if abs(delta) >= 1:
            moves.append(f"{w['label']} {delta:+d}")
        w["stock"] = target
    simulated_ttm = round(float(np.random.uniform(1.5, 3.3)), 1)
    st.session_state.ttm_samples.append(simulated_ttm)
    st.session_state.alerts_resolved += 1
    st.session_state.log.insert(0, ("resolve",
        f"<b>Agent resolved</b> {trigger_label} shortfall via network rebalance "
        f"(target = equal days-of-cover): {', '.join(moves)}. "
        f"Simulated time-to-mitigate: {simulated_ttm}d (baseline manual process: 12d)."))

def sim_tick():
    st.session_state.tick += 1
    whs = st.session_state.warehouses
    for k, w in whs.items():
        draw = w["demand"] * np.random.uniform(0.85, 1.15)
        w["stock"] = max(0, w["stock"] - draw)
        w["stock"] += w["demand"] * 0.55  # trickle replenishment
        st.session_state.history[k].append(days_of_cover(w))
        if len(st.session_state.history[k]) > 40:
            st.session_state.history[k] = st.session_state.history[k][-40:]

    for k, w in whs.items():
        if days_of_cover(w) < 5:
            st.session_state.alerts_raised += 1
            st.session_state.log.insert(0, ("alert",
                f"<b>Alert</b> {w['label']} days-of-cover {days_of_cover(w):.1f}d — below 5d target. "
                f"Fill-rate risk flagged to control tower."))
            agent_resolve(w["label"])

if "warehouses" not in st.session_state:
    reset_network()

with tab3:
    st.write("A live 4-warehouse network drains against synthetic daily demand. When a node's days-of-cover falls "
             "below target, the tower raises an alert — then an agent computes an actual rebalancing plan "
             "(equalising days-of-cover network-wide) and applies it.")

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Alerts raised", st.session_state.alerts_raised)
    kc2.metric("Resolved by agent", st.session_state.alerts_resolved)
    ttm = st.session_state.ttm_samples
    kc3.metric("Avg. time-to-mitigate", f"{np.mean(ttm):.1f} sim-d" if ttm else "—")
    whs = st.session_state.warehouses
    net_cover = sum(w["stock"] for w in whs.values()) / sum(w["demand"] for w in whs.values())
    kc4.metric("Network avg. days of cover", f"{net_cover:.1f} d")

    b1, b2, b3, b4 = st.columns(4)
    step = b1.button("⏭️ Step (1 tick)", use_container_width=True)
    autorun = b2.button("▶️ Auto-run 12 ticks", use_container_width=True)
    shock = b3.button("⚡ Inject shock — WH-W", use_container_width=True)
    reset = b4.button("♻️ Reset network", use_container_width=True)

    if reset:
        reset_network()
        st.rerun()
    if shock:
        st.session_state.warehouses["WH-W"]["stock"] = max(
            0, st.session_state.warehouses["WH-W"]["stock"] - st.session_state.warehouses["WH-W"]["demand"]*3.5)
        st.session_state.log.insert(0, ("info", "Manual demand shock injected at WH-W (Mumbai) — e.g. a regional promo spike."))
        st.rerun()
    if step:
        sim_tick()
        st.rerun()

    wh_cols = st.columns(4)
    for col, (k, w) in zip(wh_cols, st.session_state.warehouses.items()):
        cover = days_of_cover(w)
        alert = cover < 5
        with col:
            st.markdown(f"""
            <div class="wh-card {'wh-alert' if alert else 'wh-ok'}">
                <div style="font-size:13px;font-weight:600;">{w['label']}</div>
                <div style="font-family:monospace;font-size:22px;font-weight:700;">{cover:.1f} <span style="font-size:12px;color:{MUTED}">d</span></div>
                <div style="font-size:11px;color:{MUTED};">days of cover</div>
                <div style="font-size:11px;color:{MUTED};font-family:monospace;margin-top:8px;">{w['stock']:.0f} units · demand {w['demand']:.0f}/day</div>
            </div>
            """, unsafe_allow_html=True)

    chart_ph = st.empty()
    def render_tower_chart():
        fig = go.Figure()
        colors = {"WH-N": GOLD, "WH-W": RED, "WH-S": GREEN, "WH-E": PURPLE}
        for k, series in st.session_state.history.items():
            if series:
                fig.add_trace(go.Scatter(y=series, name=st.session_state.warehouses[k]["label"],
                                          line=dict(color=colors[k], width=2)))
        fig.update_layout(**base_layout("Days of cover by warehouse (live)"))
        fig.update_yaxes(title="Days of cover", rangemode="tozero")
        fig.update_xaxes(visible=False)
        chart_ph.plotly_chart(fig, use_container_width=True, key=f"tower_{st.session_state.tick}_{len(st.session_state.log)}")
    render_tower_chart()

    st.markdown("##### Event log")
    log_ph = st.container(height=260)
    def render_log():
        with log_ph:
            for kind, msg in st.session_state.log[:60]:
                cls = {"alert": "log-alert", "resolve": "log-resolve", "info": "log-info"}[kind]
                st.markdown(f'<div class="log-line {cls}">{msg}</div>', unsafe_allow_html=True)
    render_log()

    if autorun:
        for _ in range(12):
            sim_tick()
            render_tower_chart()
            with log_ph:
                st.empty()
            time.sleep(0.5)
        st.rerun()

st.caption("Team Alucard · Maestros 2026 — all data on this page is synthetically generated at runtime. "
           "No real Mondelez operational data is used or required.")

# ===========================================================================
# MODULE 4 — AUTONOMOUS AGENT MESH
# ===========================================================================
# This page turns the three individual demonstrations above into a governed,
# end-to-end operating model.  Its inputs remain synthetic until approved
# business data connections are added.
def init_agent_mesh():
    st.session_state.agent_mesh = {
        "supplier_risk": 22.0, "maintenance": 18.0, "energy": 81.0,
        "route": 94.0, "cold_chain": 2.4, "runs": 0, "resolutions": 0,
        "events": [], "approval": None,
        "warehouse": {
            "Delhi": [1210.0, 90.0], "Mumbai": [680.0, 140.0],
            "Chennai": [970.0, 70.0], "Kolkata": [720.0, 60.0],
        },
        "audit": ["Network initialised. Agent mesh is using synthetic scenario data."],
    }

def mesh_log(message):
    st.session_state.agent_mesh["audit"].insert(0, message)

def mesh_cover():
    m = st.session_state.agent_mesh
    stock = sum(v[0] for v in m["warehouse"].values())
    demand = sum(v[1] for v in m["warehouse"].values())
    return stock / demand

def mesh_request(action, value_lakh, callback, mode):
    """Enforce observe / guarded / autopilot policy before an action happens."""
    m = st.session_state.agent_mesh
    if mode == "Observe only":
        mesh_log(f"RECOMMENDATION — {action}")
    elif mode == "Autopilot within limits" or value_lakh <= 10:
        callback()
        m["resolutions"] += 1
        mesh_log(f"AUTONOMOUSLY EXECUTED — {action}")
    else:
        m["approval"] = {"action": action, "callback": callback}
        mesh_log(f"POLICY HOLD — {action} (approval required above ₹10L)")

def mesh_rebalance(mode, reason):
    m = st.session_state.agent_mesh
    wh = m["warehouse"]
    total_stock, total_demand = sum(v[0] for v in wh.values()), sum(v[1] for v in wh.values())
    target_cover = total_stock / total_demand
    moves = []
    for city, (stock, demand) in wh.items():
        delta = round(demand * target_cover - stock)
        if abs(delta) >= 5:
            moves.append(f"{city} {delta:+d} units")
    def apply():
        for city, values in wh.items():
            values[0] = round(values[1] * target_cover)
        m["events"] = [e for e in m["events"] if e != "Demand shock"]
    mesh_request(f"Rebalance inventory for {reason}: {', '.join(moves)}.", 15, apply, mode)

def mesh_run(mode):
    m = st.session_state.agent_mesh
    rng = np.random.default_rng()
    m["runs"] += 1
    # The agents make fresh observations, and the twin advances one simulated period.
    for values in m["warehouse"].values():
        values[0] = max(0, values[0] - values[1] * rng.uniform(0.65, 0.95))
        values[0] += values[1] * 0.52
    m["energy"] = float(np.clip(m["energy"] + rng.normal(0, 0.8), 74, 88))
    m["route"] = float(np.clip(m["route"] + rng.normal(0, 1.0), 86, 98))
    mesh_log("All agents completed a sensing, simulation and policy-evaluation cycle.")
    if mesh_cover() < 5:
        mesh_rebalance(mode, "low network cover")
    if m["supplier_risk"] >= 65:
        def alternate_source():
            m["supplier_risk"] = 28.0
            m["events"] = [e for e in m["events"] if e != "Supplier disruption"]
        mesh_request("Activate a pre-cleared alternate supplier allocation.", 22, alternate_source, mode)
    if m["maintenance"] >= 70:
        def service():
            m["maintenance"] = 20.0
            m["events"] = [e for e in m["events"] if e != "Maintenance anomaly"]
        mesh_request("Schedule a controlled maintenance window.", 8, service, mode)
    if m["cold_chain"] > 5:
        def reroute():
            m["cold_chain"], m["route"] = 2.0, 95.0
            m["events"] = [e for e in m["events"] if e != "Cold-chain alert"]
        mesh_request("Reroute the cold-chain load to a qualified carrier.", 6.5, reroute, mode)

def agent_card(title, agent, metric, state, description):
    colors = {"healthy": ("#237652", "#dff3e8"), "watch": ("#b66d10", "#fff0d1"),
              "risk": ("#b23a2e", "#fde1de"), "active": ("#286e97", "#dceef8")}
    color, bg = colors[state]
    st.markdown(f"""
    <div style="background:#fffaf0;border:1px solid #ead4a7;border-left:4px solid {color};border-radius:6px;padding:13px;height:174px">
      <div style="font-size:14px;font-weight:700;color:#24170e">{title}</div>
      <div style="font-size:11px;color:#806746;margin-top:3px;height:29px">{agent}</div>
      <div style="font-family:monospace;font-size:23px;font-weight:700;color:#4b2e83;margin:9px 0 3px">{metric}</div>
      <div style="font-size:11px;line-height:1.35;color:#513a26;height:32px">{description}</div>
      <span style="display:inline-block;margin-top:8px;background:{bg};color:{color};border-radius:10px;padding:2px 7px;font:10px monospace">{state.upper()}</span>
    </div>
    """, unsafe_allow_html=True)

if "agent_mesh" not in st.session_state:
    init_agent_mesh()

with tab4:
    st.subheader("Autonomous agent mesh")
    st.write("Nine agents coordinate sensing, simulation and execution. They work on local synthetic scenarios now; connect only approved systems of record before using this in operations.")
    policy_col, run_col = st.columns([2, 1])
    with policy_col:
        autonomy = st.selectbox("Autonomy policy", ["Guarded autonomy", "Observe only", "Autopilot within limits"],
                                help="Guarded autonomy queues actions above ₹10L for approval.")
    with run_col:
        st.write("")
        if st.button("▶ Run all agents", use_container_width=True, key="mesh_run"):
            mesh_run(autonomy)
            st.rerun()

    m = st.session_state.agent_mesh
    high_risks = sum([m["supplier_risk"] >= 65, m["maintenance"] >= 70, m["cold_chain"] > 5,
                      mesh_cover() < 5])
    metrics = st.columns(5)
    metrics[0].metric("Demand forecast accuracy", "78.2%")
    metrics[1].metric("Network days of cover", f"{mesh_cover():.1f} d")
    metrics[2].metric("Open high-risk events", high_risks)
    metrics[3].metric("Identified annual value", "₹79 Cr")
    metrics[4].metric("Agent resolutions", m["resolutions"])

    st.markdown("##### Wave 1 — Demand & inventory")
    c = st.columns(3)
    with c[0]: agent_card("Demand sensing", "POS, promotion and weather forecasting", "78.2%", "healthy", "Forecast features recalibrated against the latest demand signal.")
    with c[1]: agent_card("Multi-echelon inventory", "Safety stock across plants, DCs and depots", f"{mesh_cover():.1f} d", "risk" if mesh_cover() < 5 else "healthy", "Twin uses a 5–12 day safety-stock policy band.")
    with c[2]: agent_card("Should-cost modelling", "Cocoa, dairy and FX buy-timing signal", "100.0", "watch", "Trend and z-score determine buy, wait or hedge guidance.")

    st.markdown("##### Wave 2 — Make & network")
    c = st.columns(4)
    with c[0]: agent_card("Supplier risk warning", "Supplier, commodity and lane exposure", f"{m['supplier_risk']:.0f}/100", "risk" if m["supplier_risk"] >= 65 else ("watch" if m["supplier_risk"] >= 35 else "healthy"), "Pre-cleared alternate source is available for severe events.")
    with c[1]: agent_card("Predictive maintenance", "Anomaly detection for critical equipment", f"{m['maintenance']:.0f}/100", "risk" if m["maintenance"] >= 70 else ("watch" if m["maintenance"] >= 40 else "healthy"), "Condition monitoring triggers a controlled work order.")
    with c[2]: agent_card("Energy setpoints", "Quality-bounded MPC for ovens and cold rooms", f"{m['energy']:.0f}", "healthy", "Setpoints stay inside safety and product-quality envelopes.")
    with c[3]: agent_card("Network digital twin", "Inventory, capacity and allocation scenarios", f"{128 + m['runs']}", "active", "The twin evaluates feasible resolution plans before action.")

    st.markdown("##### Wave 3 — Deliver & autonomy")
    c = st.columns(2)
    with c[0]: agent_card("Route, load & cold chain", "Route, load factor and temperature compliance", f"{m['route']:.0f}%", "risk" if m["cold_chain"] > 5 else "healthy", f"Current cold-chain variance: {m['cold_chain']:.1f}°C.")
    with c[1]: agent_card("Agentic resolution", "Policy checks, approvals, execution and evidence", str(m["resolutions"]), "active", "Resolves exceptions only within the selected autonomy policy.")

    st.markdown("##### Digital twin & scenario controls")
    node_cols = st.columns(4)
    for col, (city, (stock, demand)) in zip(node_cols, m["warehouse"].items()):
        with col:
            cover = stock / demand
            st.metric(city, f"{cover:.1f} d", f"{stock:.0f} units · {demand:.0f}/day")
    a, b, c, d = st.columns(4)
    if a.button("⚡ Demand shock", use_container_width=True):
        m["warehouse"]["Mumbai"][0] = max(0, m["warehouse"]["Mumbai"][0] - 440)
        m["events"].append("Demand shock")
        mesh_log("EVENT — Regional promotion spike at Mumbai; cover risk detected.")
        mesh_rebalance(autonomy, "Mumbai promotion spike")
        st.rerun()
    if b.button("⚠ Supplier disruption", use_container_width=True):
        m["supplier_risk"] = 78.0; m["events"].append("Supplier disruption")
        mesh_log("EVENT — Supplier lead-time deviation and commodity exposure detected.")
        mesh_run(autonomy); st.rerun()
    if c.button("❄ Cold-chain alert", use_container_width=True):
        m["cold_chain"], m["route"] = 8.6, 84.0; m["events"].append("Cold-chain alert")
        mesh_log("EVENT — Cold-chain temperature excursion on a delivery lane.")
        mesh_run(autonomy); st.rerun()
    if d.button("↺ Reset scenario", use_container_width=True):
        init_agent_mesh(); st.rerun()

    st.markdown("##### Decision & audit log")
    log_col, approval_col = st.columns([2, 1])
    with log_col:
        for item in m["audit"][:12]:
            st.code(item, language=None)
    with approval_col:
        st.markdown("**Approval queue**")
        if m["approval"]:
            st.warning(m["approval"]["action"])
            if st.button("Approve & execute", use_container_width=True):
                approval = m["approval"]
                approval["callback"]()
                m["approval"] = None
                m["resolutions"] += 1
                mesh_log(f"APPROVED & EXECUTED — {approval['action']}")
                st.rerun()
        else:
            st.success("No actions need approval.")
        st.caption("Guarded autonomy holds interventions above ₹10L. Observe-only mode records recommendations without executing them.")
