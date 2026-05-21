"""
╔══════════════════════════════════════════════════════════════════╗
║       UPSTOX ALPHA TRADING ENGINE v2 — Live Options Matrix      ║
║  Tabbed layout · Plotly charts · IV Percentile · P&L Heatmap   ║
║  OI Change Tracking · PCR Trend · Interactive Strategy Builder  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
from scipy.stats import norm
import requests
import urllib.parse
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import colorsys

# ═══════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="Upstox Alpha Engine v2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
#  THEME CSS
# ═══════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --bg-card-alt: #1a2236;
    --border: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --accent-green: #22c55e;
    --accent-red: #ef4444;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-amber: #f59e0b;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    overflow: visible !important;
}
div[data-testid="stMetric"] label {
    color: #cbd5e1 !important;
    font-size: 10px !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800 !important;
    font-size: 20px !important;
    color: #ffffff !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] svg {
    display: none;
}

/* Tabs */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px;
}

/* Sentiment card */
.signal-card {
    border-radius: 12px; padding: 20px; margin: 12px 0;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid rgba(255,255,255,0.08);
}
.signal-label { font-size: 10px; letter-spacing: 2.5px; color: rgba(255,255,255,0.7); margin-bottom: 2px; text-transform: uppercase; }
.signal-value { font-size: 26px; font-weight: 700; color: #fff; }
.signal-sub { font-size: 12px; color: rgba(255,255,255,0.75); margin-top: 6px; line-height: 1.6; }

/* Strategy card */
.strat-card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px; margin: 10px 0; color: #ffffff;
}
.strat-title {
    font-size: 14px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border);
}
.strat-leg {
    font-size: 14px; line-height: 2.0; font-family: 'JetBrains Mono', monospace;
    color: #f1f5f9 !important; font-weight: 600;
}
.strat-leg b { color: #ffffff; font-weight: 800; }
.strat-profit { font-size: 18px; font-weight: 800; margin-top: 12px; }

/* Badge pills */
.badge { display: inline-block; padding: 3px 10px; border-radius: 5px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace; margin: 2px 3px; }
.badge-green { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-red { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-blue { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.badge-purple { background: rgba(139,92,246,0.15); color: #a78bfa; border: 1px solid rgba(139,92,246,0.3); }
.badge-amber { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }

/* IV Gauge */
.iv-gauge-container { display: flex; align-items: center; gap: 12px; margin: 6px 0; }
.iv-gauge-bar { flex: 1; height: 10px; border-radius: 5px; background: linear-gradient(90deg, #22c55e 0%, #f59e0b 50%, #ef4444 100%); position: relative; }
.iv-gauge-marker { position: absolute; top: -4px; width: 4px; height: 18px; background: #fff; border-radius: 2px; transform: translateX(-50%); box-shadow: 0 0 6px rgba(255,255,255,0.5); }
.iv-pct-label { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; min-width: 60px; text-align: right; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  CONSTANTS & INDEX DEFINITIONS
# ═══════════════════════════════════════════════

UPSTOX_BASE = "https://api.upstox.com/v2"

INDICES = {
    "NIFTY 50":     {"key": "NSE_INDEX|Nifty 50",          "symbol": "NIFTY",       "diff": 50,  "lot": 25},
    "BANK NIFTY":   {"key": "NSE_INDEX|Nifty Bank",        "symbol": "BANKNIFTY",   "diff": 100, "lot": 15},
    "FINNIFTY":     {"key": "NSE_INDEX|Nifty Fin Service",  "symbol": "FINNIFTY",    "diff": 50,  "lot": 25},
    "MIDCAP NIFTY": {"key": "NSE_INDEX|NIFTY MID SELECT",  "symbol": "MIDCPNIFTY",  "diff": 25,  "lot": 50},
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="Inter, sans-serif", size=11, color="#94a3b8"),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    xaxis=dict(gridcolor="rgba(30,41,59,0.6)", zeroline=False),
    yaxis=dict(gridcolor="rgba(30,41,59,0.6)", zeroline=False),
)

# ═══════════════════════════════════════════════
#  UPSTOX API CLIENT
# ═══════════════════════════════════════════════

class UpstoxClient:
    def __init__(self, token: str):
        clean = token.strip().replace("Bearer ", "")
        self.headers = {"Authorization": f"Bearer {clean}", "Accept": "application/json"}

    def _safe_json(self, r):
        ct = r.headers.get("Content-Type", "").lower()
        if "application/json" not in ct:
            raise ValueError(f"Non-JSON response (token expired?). Status {r.status_code}: {r.text[:200]}")
        body = r.json()
        if body.get("status") == "error":
            errs = body.get("errors", [])
            msg = errs[0].get("message", str(errs)) if errs else str(body)
            raise ValueError(f"Upstox API: {msg}")
        return body

    def get_spot_price(self, instrument_key: str) -> float:
        r = requests.get(f"{UPSTOX_BASE}/market-quote/ltp",
                         headers=self.headers, params={"instrument_key": instrument_key}, timeout=10)
        r.raise_for_status()
        data = self._safe_json(r).get("data", {})
        for k, v in data.items():
            if k == instrument_key or k.lower().replace(" ", "") == instrument_key.lower().replace(" ", ""):
                return float(v["last_price"])
        first = next(iter(data.values()), None)
        if first:
            return float(first["last_price"])
        raise ValueError(f"Symbol not found: {instrument_key}")

    def get_expiries(self, instrument_key: str) -> list:
        r = requests.get(f"{UPSTOX_BASE}/option/contract",
                         headers=self.headers, params={"instrument_key": instrument_key}, timeout=10)
        r.raise_for_status()
        data = self._safe_json(r).get("data", [])
        expiries = sorted(set(
            str(c.get("expiry", ""))[:10] for c in data
        ))
        return [e for e in expiries if e and e != "None"]

    def get_option_chain(self, instrument_key: str, expiry_date: str) -> list:
        r = requests.get(f"{UPSTOX_BASE}/option/chain",
                         headers=self.headers,
                         params={"instrument_key": instrument_key, "expiry_date": expiry_date}, timeout=10)
        r.raise_for_status()
        return self._safe_json(r).get("data", [])

    def get_historical_candles(self, instrument_key: str, interval="day", days=45) -> pd.DataFrame:
        to_d = datetime.now().strftime("%Y-%m-%d")
        from_d = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        enc = urllib.parse.quote(instrument_key, safe="")
        r = requests.get(f"{UPSTOX_BASE}/historical-candle/{enc}/{interval}/{to_d}/{from_d}",
                         headers=self.headers, timeout=10)
        r.raise_for_status()
        candles = self._safe_json(r).get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()
        rows = [{"ts": c[0], "open": float(c[1]), "high": float(c[2]),
                 "low": float(c[3]), "close": float(c[4]),
                 "volume": int(c[5]) if len(c) > 5 else 0} for c in candles if len(c) >= 5]
        return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


# ═══════════════════════════════════════════════
#  ANALYTICS FUNCTIONS
# ═══════════════════════════════════════════════

def compute_adx(df: pd.DataFrame, period=14):
    if df.empty or len(df) < period * 2 + 2:
        return None
    d = df.copy()
    d["ph"], d["pl"], d["pc"] = d["high"].shift(1), d["low"].shift(1), d["close"].shift(1)
    d["tr"] = d.apply(lambda r: max(r["high"]-r["low"],
        abs(r["high"]-r["pc"]) if pd.notna(r["pc"]) else 0,
        abs(r["low"]-r["pc"]) if pd.notna(r["pc"]) else 0), axis=1)
    d["+dm"] = d.apply(lambda r: max(r["high"]-r["ph"],0)
        if pd.notna(r["ph"]) and (r["high"]-r["ph"])>(r["pl"]-r["low"]) else 0, axis=1)
    d["-dm"] = d.apply(lambda r: max(r["pl"]-r["low"],0)
        if pd.notna(r["pl"]) and (r["pl"]-r["low"])>(r["high"]-r["ph"]) else 0, axis=1)
    d = d.iloc[1:].reset_index(drop=True)
    tr_s = [d["tr"].iloc[:period].sum()]
    pd_s = [d["+dm"].iloc[:period].sum()]
    nd_s = [d["-dm"].iloc[:period].sum()]
    for i in range(period, len(d)):
        tr_s.append(tr_s[-1] - tr_s[-1]/period + d["tr"].iloc[i])
        pd_s.append(pd_s[-1] - pd_s[-1]/period + d["+dm"].iloc[i])
        nd_s.append(nd_s[-1] - nd_s[-1]/period + d["-dm"].iloc[i])
    pdi_l, ndi_l, dx_l = [], [], []
    for i in range(len(tr_s)):
        pdi = pd_s[i]/tr_s[i]*100 if tr_s[i]>0 else 0
        ndi = nd_s[i]/tr_s[i]*100 if tr_s[i]>0 else 0
        pdi_l.append(pdi); ndi_l.append(ndi)
        dx_l.append(abs(pdi-ndi)/(pdi+ndi)*100 if (pdi+ndi)>0 else 0)
    if len(dx_l) < period:
        return None
    adx_l = [sum(dx_l[:period])/period]
    for i in range(period, len(dx_l)):
        adx_l.append((adx_l[-1]*(period-1)+dx_l[i])/period)
    return {"adx": round(adx_l[-1],2), "plus_di": round(pdi_l[-1],2), "minus_di": round(ndi_l[-1],2)}


def bs_greeks(S, K, T, r, sigma, opt="CE"):
    if T <= 0 or sigma <= 0:
        return {"price":0,"delta":0,"gamma":0,"theta":0,"vega":0}
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)/100
    if opt == "CE":
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1)-1
        theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    return {"price":round(price,2),"delta":round(delta,3),"gamma":round(gamma,5),"theta":round(theta,2),"vega":round(vega,2)}


def compute_max_pain(df_chain: pd.DataFrame) -> float:
    strikes = df_chain["Strike"].values
    pain = {}
    for s in strikes:
        total = 0.0
        for _, row in df_chain.iterrows():
            total += max(0, s - row["Strike"]) * row["PE OI"]
            total += max(0, row["Strike"] - s) * row["CE OI"]
        pain[s] = total
    return min(pain, key=pain.get) if pain else 0.0


def compute_iv_percentile(candles_df: pd.DataFrame, current_iv: float, window=30) -> float:
    """IV Percentile: % of last N days where realized vol was below current IV."""
    if candles_df.empty or len(candles_df) < window + 2:
        return 50.0  # default mid
    closes = candles_df["close"].values
    log_ret = np.diff(np.log(closes))
    if len(log_ret) < window:
        return 50.0
    # Rolling realized vol (annualised)
    rv_series = []
    for i in range(len(log_ret) - window + 1):
        chunk = log_ret[i:i+window]
        rv = np.std(chunk) * np.sqrt(252) * 100
        rv_series.append(rv)
    if not rv_series:
        return 50.0
    count_below = sum(1 for rv in rv_series if rv < current_iv * 100)
    return round(count_below / len(rv_series) * 100, 1)


def compute_pnl_heatmap(strategy_legs, lot_size, spot_price, diff):
    """
    Compute strategy P&L across a grid of spot prices and days-to-expiry.
    Each leg: {"strike": K, "type": "CE"/"PE", "action": "BUY"/"SELL", "premium": ltp}
    """
    spot_range = np.linspace(spot_price - 8*diff, spot_price + 8*diff, 50)
    dte_range = np.array([0, 1, 2, 3, 5, 7, 10, 14])  # days to expiry

    pnl_matrix = np.zeros((len(dte_range), len(spot_range)))

    for i, dte in enumerate(dte_range):
        for j, s in enumerate(spot_range):
            total = 0.0
            for leg in strategy_legs:
                K = leg["strike"]
                prem = leg["premium"]
                if dte == 0:
                    # At expiry — intrinsic only
                    if leg["type"] == "CE":
                        val = max(0, s - K)
                    else:
                        val = max(0, K - s)
                else:
                    T = dte / 365
                    val = bs_greeks(s, K, T, 0.07, 0.15, leg["type"])["price"]

                if leg["action"] == "BUY":
                    total += (val - prem)
                else:
                    total += (prem - val)
            pnl_matrix[i, j] = total * lot_size
    return spot_range, dte_range, pnl_matrix


# ═══════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════

st.sidebar.markdown("### 🔐 Authentication")
api_token = st.sidebar.text_input("Access Token", type="password", value="",
                                   help="Paste your Upstox OAuth access_token. Expires midnight IST daily.")

st.sidebar.markdown("---")
selected_index_name = st.sidebar.selectbox("🎯 Underlying Index", list(INDICES.keys()))
index_meta = INDICES[selected_index_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Parameters")
iv_override = st.sidebar.slider("Manual IV Override (%)", 5.0, 80.0, 15.0, 0.5) / 100
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 12.0, 7.0, 0.1) / 100
strike_depth = st.sidebar.slider("Strike Depth Around ATM", 3, 15, 7)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Auto-Refresh", value=False)
refresh_interval = 30
if auto_refresh:
    refresh_interval = st.sidebar.slider("Interval (sec)", 10, 120, 30, 5)

# ═══════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════

st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
    <span style="font-size:28px;">⚡</span>
    <span style="font-family:'Inter',sans-serif; font-size:22px; font-weight:700;">
        Upstox Alpha Engine
    </span>
    <span style="font-size:12px; color:#64748b; background:rgba(59,130,246,0.12); padding:2px 8px;
                 border-radius:4px; font-weight:600; margin-left:4px;">v2.0</span>
    <span style="font-size:13px; color:#94a3b8; margin-left:auto; font-family:'JetBrains Mono',monospace;">
        {selected_index_name}
    </span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  MAIN ENGINE
# ═══════════════════════════════════════════════

if not api_token:
    st.info("💡 Enter your **Upstox Access Token** in the sidebar to begin. "
            "Get it via the Upstox OAuth2 flow — it resets at midnight IST.")
    st.stop()

try:
    client = UpstoxClient(token=api_token)
    diff = index_meta["diff"]
    lot_size = index_meta["lot"]

    spot_price = client.get_spot_price(index_meta["key"])
    expiries = client.get_expiries(index_meta["key"])

    if not expiries:
        st.error("No active derivative contracts found.")
        st.stop()

    selected_expiry = st.sidebar.selectbox("📅 Expiry", expiries, index=0)

    expiry_dt = datetime.strptime(selected_expiry, "%Y-%m-%d").replace(hour=15, minute=30)
    tte_years = max((expiry_dt - datetime.now()).total_seconds() / (86400*365), 0.0001)
    tte_days = max(tte_years * 365, 0.01)

    with st.spinner("Loading chain & candles..."):
        candles_df = client.get_historical_candles(index_meta["key"], "day", 60)
        adx_metrics = compute_adx(candles_df)
        chain_raw = client.get_option_chain(index_meta["key"], selected_expiry)

    if not chain_raw:
        st.warning("Empty chain for this expiry.")
        st.stop()

    atm_strike = round(spot_price / diff) * diff

    # ── Build chain records ──
    records = []
    for sd in chain_raw:
        sp = float(sd.get("strike_price", 0))
        if abs(sp - atm_strike) > strike_depth * diff:
            continue
        ce = sd.get("call_options", {}) or {}
        pe = sd.get("put_options", {}) or {}
        ce_md = ce.get("market_data", {}) or {}
        pe_md = pe.get("market_data", {}) or {}

        ce_oi = ce_md.get("oi", 0)
        pe_oi = pe_md.get("oi", 0)
        ce_ltp = ce_md.get("ltp", 0)
        pe_ltp = pe_md.get("ltp", 0)
        ce_vol = ce_md.get("volume", 0)
        pe_vol = pe_md.get("volume", 0)
        ce_prev_oi = ce_md.get("prev_oi", ce_oi)  # fallback if not available
        pe_prev_oi = pe_md.get("prev_oi", pe_oi)
        ce_iv_raw = ce_md.get("iv", 0)
        pe_iv_raw = pe_md.get("iv", 0)

        ce_sigma = (ce_iv_raw/100) if ce_iv_raw and ce_iv_raw > 0 else iv_override
        pe_sigma = (pe_iv_raw/100) if pe_iv_raw and pe_iv_raw > 0 else iv_override

        ce_g = bs_greeks(spot_price, sp, tte_years, risk_free_rate, ce_sigma, "CE")
        pe_g = bs_greeks(spot_price, sp, tte_years, risk_free_rate, pe_sigma, "PE")

        records.append({
            "CE OI": ce_oi, "CE OI Chg": ce_oi - ce_prev_oi, "CE Vol": ce_vol,
            "CE IV": round(ce_sigma*100, 1),
            "CE Delta": ce_g["delta"], "CE Gamma": ce_g["gamma"],
            "CE Theta": ce_g["theta"], "CE Vega": ce_g["vega"],
            "CE LTP": ce_ltp,
            "Strike": sp,
            "PE LTP": pe_ltp,
            "PE Vega": pe_g["vega"], "PE Theta": pe_g["theta"],
            "PE Gamma": pe_g["gamma"], "PE Delta": pe_g["delta"],
            "PE IV": round(pe_sigma*100, 1),
            "PE Vol": pe_vol, "PE OI Chg": pe_oi - pe_prev_oi, "PE OI": pe_oi,
        })

    df = pd.DataFrame(records).sort_values("Strike").reset_index(drop=True)

    # ── Derived metrics ──
    total_ce_oi = df["CE OI"].sum()
    total_pe_oi = df["PE OI"].sum()
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
    max_pain = compute_max_pain(df)
    resistance = df.loc[df["CE OI"].idxmax(), "Strike"] if not df.empty else atm_strike
    support = df.loc[df["PE OI"].idxmax(), "Strike"] if not df.empty else atm_strike

    # IV Percentile
    atm_row = df[df["Strike"] == atm_strike]
    atm_iv = float(atm_row.iloc[0]["CE IV"]) if not atm_row.empty else iv_override * 100
    iv_pct = compute_iv_percentile(candles_df, atm_iv / 100, window=20)

    # PCR history in session state for sparkline
    if "pcr_history" not in st.session_state:
        st.session_state.pcr_history = []
    st.session_state.pcr_history.append({"time": datetime.now().strftime("%H:%M:%S"), "pcr": pcr})
    if len(st.session_state.pcr_history) > 60:
        st.session_state.pcr_history = st.session_state.pcr_history[-60:]

    # Sentiment
    if pcr >= 1.25:
        sentiment, sent_color = "STRONG BULLISH", "#15803d"
        card_bg = "linear-gradient(135deg, #15803d, #166534)"
    elif pcr > 1.05:
        sentiment, sent_color = "MILDLY BULLISH", "#22c55e"
        card_bg = "linear-gradient(135deg, #065f46, #064e3b)"
    elif pcr <= 0.75:
        sentiment, sent_color = "STRONG BEARISH", "#dc2626"
        card_bg = "linear-gradient(135deg, #991b1b, #7f1d1d)"
    elif pcr < 0.95:
        sentiment, sent_color = "MILDLY BEARISH", "#ef4444"
        card_bg = "linear-gradient(135deg, #b91c1c, #991b1b)"
    else:
        sentiment, sent_color = "NEUTRAL", "#64748b"
        card_bg = "linear-gradient(135deg, #475569, #334155)"

    # σ bounds
    std_price = spot_price * iv_override * np.sqrt(tte_years)
    lo1 = spot_price - std_price
    hi1 = spot_price + std_price
    lo2 = spot_price - 2*std_price
    hi2 = spot_price + 2*std_price

    # Strategy strikes
    ic_sell_put = round(lo1 / diff) * diff
    ic_sell_call = round(hi1 / diff) * diff
    ic_buy_put = ic_sell_put - diff
    ic_buy_call = ic_sell_call + diff

    # ═══════════════════════════════════════════════
    #  TOP KPI ROW
    # ═══════════════════════════════════════════════

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Spot", f"₹{spot_price:,.2f}")
    k2.metric("ATM", f"{atm_strike:,.0f}")
    k3.metric("PCR", f"{pcr}")
    k4.metric("Max Pain", f"{max_pain:,.0f}")
    adx_val = adx_metrics["adx"] if adx_metrics else None
    k5.metric("ADX (14)", f"{adx_val}" if adx_val else "—",
              f"+DI {adx_metrics['plus_di']}/-DI {adx_metrics['minus_di']}" if adx_metrics else None)
    k6.metric("DTE", f"{tte_days:.1f} days")

    # ── IV Percentile Gauge ──
    iv_color = "#22c55e" if iv_pct < 30 else "#f59e0b" if iv_pct < 70 else "#ef4444"
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:14px; margin:8px 0 4px 0;">
        <span style="font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; min-width:110px;">IV Percentile</span>
        <div class="iv-gauge-bar" style="flex:1;">
            <div class="iv-gauge-marker" style="left:{min(iv_pct, 100)}%;"></div>
        </div>
        <span class="iv-pct-label" style="color:{iv_color};">{iv_pct:.0f}%</span>
        <span style="font-size:11px; color:#64748b;">ATM IV: {atm_iv:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sentiment Card ──
    st.markdown(f"""
    <div class="signal-card" style="background:{card_bg};">
        <div class="signal-label">Structural Trend Signal</div>
        <div class="signal-value">{sentiment}</div>
        <div class="signal-sub">
            Put OI: {total_pe_oi:,.0f} &nbsp;·&nbsp; Call OI: {total_ce_oi:,.0f}
            &nbsp;&nbsp;
            <span class="badge badge-green">Support {support:,.0f}</span>
            <span class="badge badge-red">Resistance {resistance:,.0f}</span>
            <span class="badge badge-purple">Max Pain {max_pain:,.0f}</span>
            <span class="badge badge-amber">IV Pct {iv_pct:.0f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    #  TABBED LAYOUT
    # ═══════════════════════════════════════════════

    tab_dash, tab_chain, tab_strat, tab_charts = st.tabs([
        "📊 Dashboard", "📋 Options Chain", "🛡️ Strategy Builder", "📈 Charts"
    ])

    # ──────────────────────────────────
    #  TAB 1: DASHBOARD
    # ──────────────────────────────────
    with tab_dash:

        # Normal distribution
        st.markdown("#### 🎯 Statistical Price Forecast")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("2σ Low (95.4%)", f"₹{lo2:,.0f}")
        pc2.metric("1σ Low (68.2%)", f"₹{lo1:,.0f}")
        pc3.metric("1σ High (68.2%)", f"₹{hi1:,.0f}")
        pc4.metric("2σ High (95.4%)", f"₹{hi2:,.0f}")

        # Bell curve with Plotly
        x = np.linspace(spot_price - 3.5*std_price, spot_price + 3.5*std_price, 400)
        y = norm.pdf(x, spot_price, std_price)

        fig_bell = go.Figure()
        # 2σ fill
        mask_2s = (x >= lo2) & (x <= hi2)
        fig_bell.add_trace(go.Scatter(x=x[mask_2s], y=y[mask_2s], fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)", line=dict(width=0), name="95.4% (2σ)", showlegend=True))
        # 1σ fill
        mask_1s = (x >= lo1) & (x <= hi1)
        fig_bell.add_trace(go.Scatter(x=x[mask_1s], y=y[mask_1s], fill="tozeroy",
            fillcolor="rgba(56,189,248,0.22)", line=dict(width=0), name="68.2% (1σ)", showlegend=True))
        # PDF line
        fig_bell.add_trace(go.Scatter(x=x, y=y, mode="lines",
            line=dict(color="#e2e8f0", width=2), name="PDF"))
        # Vertical lines
        for val, clr, nm, dash in [
            (spot_price, "#f1f5f9", "Spot", "solid"),
            (max_pain, "#8b5cf6", "Max Pain", "dashdot"),
            (support, "#22c55e", "Support", "dot"),
            (resistance, "#ef4444", "Resistance", "dot"),
        ]:
            fig_bell.add_vline(x=val, line=dict(color=clr, width=1.5, dash=dash),
                               annotation_text=nm, annotation_position="top")

        fig_bell.update_layout(**PLOTLY_LAYOUT, height=320,
            title=dict(text=f"Expiry Forecast — {selected_index_name} ({selected_expiry})", font=dict(size=13)),
            yaxis_title="Probability Density", xaxis_title="Settlement Price")
        st.plotly_chart(fig_bell, use_container_width=True)

        # PCR Trend sparkline
        if len(st.session_state.pcr_history) > 1:
            st.markdown("#### 📉 PCR Trend (Session)")
            pcr_df = pd.DataFrame(st.session_state.pcr_history)
            fig_pcr = go.Figure()
            fig_pcr.add_trace(go.Scatter(x=pcr_df["time"], y=pcr_df["pcr"], mode="lines+markers",
                line=dict(color="#3b82f6", width=2), marker=dict(size=4), name="PCR"))
            fig_pcr.add_hline(y=1.0, line=dict(color="#64748b", dash="dash", width=1), annotation_text="Neutral")
            fig_pcr.update_layout(**PLOTLY_LAYOUT, height=200,
                yaxis_title="PCR", xaxis_title="Time")
            st.plotly_chart(fig_pcr, use_container_width=True)

        # OI Change summary
        st.markdown("#### 🔄 OI Change (Current vs Previous)")
        oc1, oc2 = st.columns(2)
        with oc1:
            top_ce_buildup = df.nlargest(5, "CE OI Chg")[["Strike", "CE OI", "CE OI Chg"]]
            st.markdown("**Top Call OI Build-up**")
            st.dataframe(top_ce_buildup.style.format({"Strike":"{:,.0f}","CE OI":"{:,.0f}","CE OI Chg":"{:+,.0f}"}),
                         use_container_width=True, hide_index=True)
        with oc2:
            top_pe_buildup = df.nlargest(5, "PE OI Chg")[["Strike", "PE OI", "PE OI Chg"]]
            st.markdown("**Top Put OI Build-up**")
            st.dataframe(top_pe_buildup.style.format({"Strike":"{:,.0f}","PE OI":"{:,.0f}","PE OI Chg":"{:+,.0f}"}),
                         use_container_width=True, hide_index=True)

    # ──────────────────────────────────
    #  TAB 2: OPTIONS CHAIN
    # ──────────────────────────────────
    with tab_chain:
        st.markdown("#### Live Options Chain — Greeks & Market Data")

        # Build a color-mapped HTML table for better visual density
        display_cols = ["CE OI","CE OI Chg","CE Vol","CE IV","CE Delta","CE Theta","CE LTP",
                        "Strike",
                        "PE LTP","PE Theta","PE Delta","PE IV","PE Vol","PE OI Chg","PE OI"]
        df_display = df[display_cols].copy()

        def oi_bg(val, max_val):
            if max_val == 0: return ""
            intensity = min(abs(val)/max_val, 1.0)
            return f"rgba(59,130,246,{intensity*0.25})"

        def chg_color(val):
            if val > 0: return "color: #4ade80;"
            elif val < 0: return "color: #f87171;"
            return ""

        max_oi = max(df["CE OI"].max(), df["PE OI"].max(), 1)

        def style_chain(row):
            s = row["Strike"]
            styles = [""] * len(row)
            si = list(row.index)

            # ATM highlight
            if s == atm_strike:
                styles = ["background-color: rgba(245,158,11,0.15); font-weight:700;"] * len(row)

            # OI intensity on CE OI and PE OI columns
            for col_name in ["CE OI", "PE OI"]:
                idx = si.index(col_name)
                intensity = min(abs(row[col_name])/max_oi, 1.0) if max_oi > 0 else 0
                bg = f"rgba(59,130,246,{intensity*0.3})"
                styles[idx] += f" background-color: {bg};"

            # OI Change coloring
            for col_name in ["CE OI Chg", "PE OI Chg"]:
                idx = si.index(col_name)
                v = row[col_name]
                if v > 0:
                    styles[idx] += " color: #4ade80;"
                elif v < 0:
                    styles[idx] += " color: #f87171;"

            return styles

        styled = df_display.style.apply(style_chain, axis=1).format({
            "CE OI":"{:,.0f}", "CE OI Chg":"{:+,.0f}", "CE Vol":"{:,.0f}", "CE IV":"{:.1f}",
            "CE Delta":"{:.3f}", "CE Theta":"{:.2f}", "CE LTP":"₹{:.2f}",
            "Strike":"{:,.0f}",
            "PE LTP":"₹{:.2f}", "PE Theta":"{:.2f}", "PE Delta":"{:.3f}", "PE IV":"{:.1f}",
            "PE Vol":"{:,.0f}", "PE OI Chg":"{:+,.0f}", "PE OI":"{:,.0f}",
        })
        st.dataframe(styled, use_container_width=True, height=500)

    # ──────────────────────────────────
    #  TAB 3: STRATEGY BUILDER
    # ──────────────────────────────────
    with tab_strat:
        st.markdown("#### 🛡️ Strategy Playbook")

        # Recommendation
        adx_v = adx_metrics["adx"] if adx_metrics else 15
        if adx_v > 25:
            rec = "Directional: Debit Spreads / Long Options"
            rec_icon = "🚀"
        elif iv_pct > 70:
            rec = "High IV: Iron Butterfly / Short Straddle"
            rec_icon = "🦋"
        elif iv_pct < 30:
            rec = "Low IV: Long Straddle / Calendar Spread"
            rec_icon = "📈"
        else:
            rec = "Range-bound: Iron Condor"
            rec_icon = "📊"

        st.info(f"{rec_icon} **Engine Recommendation**: ADX={adx_v}, IV Pct={iv_pct:.0f}% → **{rec}**")

        strat_choice = st.selectbox("Select Strategy", ["Iron Condor", "Short Straddle", "Iron Butterfly", "Bull Put Spread", "Bear Call Spread"])

        def get_ltp(strike):
            m = df[df["Strike"] == strike]
            if not m.empty:
                return float(m.iloc[0]["CE LTP"]), float(m.iloc[0]["PE LTP"])
            return 0.0, 0.0

        legs = []

        if strat_choice == "Iron Condor":
            c_sell_ce, _ = get_ltp(ic_sell_call)
            _, p_sell_pe = get_ltp(ic_sell_put)
            c_buy_ce, _ = get_ltp(ic_buy_call)
            _, p_buy_pe = get_ltp(ic_buy_put)
            net = max((c_sell_ce + p_sell_pe) - (c_buy_ce + p_buy_pe), 0)
            max_risk = diff - net if net > 0 else diff
            legs = [
                {"strike":ic_buy_put,"type":"PE","action":"BUY","premium":p_buy_pe},
                {"strike":ic_sell_put,"type":"PE","action":"SELL","premium":p_sell_pe},
                {"strike":ic_sell_call,"type":"CE","action":"SELL","premium":c_sell_ce},
                {"strike":ic_buy_call,"type":"CE","action":"BUY","premium":c_buy_ce},
            ]
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-title" style="color:#3b82f6;">📊 Iron Condor</div>
                <div class="strat-leg">
                    BUY 1× <b>{ic_buy_put} PE</b> @ ₹{p_buy_pe:.2f}<br>
                    SELL 1× <b>{ic_sell_put} PE</b> @ ₹{p_sell_pe:.2f}<br>
                    SELL 1× <b>{ic_sell_call} CE</b> @ ₹{c_sell_ce:.2f}<br>
                    BUY 1× <b>{ic_buy_call} CE</b> @ ₹{c_buy_ce:.2f}
                </div>
                <div class="strat-profit" style="color:#4ade80;">
                    💰 Net Credit: ₹{net:,.2f}/lot &nbsp;(₹{net*lot_size:,.0f} total)
                </div>
                <div style="color:#fca5a5; font-size:14px; margin-top:6px; font-weight:700;">
                    ⚠️ Max Risk: ₹{max_risk:,.2f}/lot &nbsp;(₹{max_risk*lot_size:,.0f} total)
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif strat_choice == "Short Straddle":
            c_atm, p_atm = get_ltp(atm_strike)
            net = c_atm + p_atm
            upper_be = atm_strike + net
            lower_be = atm_strike - net
            legs = [
                {"strike":atm_strike,"type":"CE","action":"SELL","premium":c_atm},
                {"strike":atm_strike,"type":"PE","action":"SELL","premium":p_atm},
            ]
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-title" style="color:#f59e0b;">🔥 Short Straddle</div>
                <div class="strat-leg">
                    SELL 1× <b>{atm_strike} CE</b> @ ₹{c_atm:.2f}<br>
                    SELL 1× <b>{atm_strike} PE</b> @ ₹{p_atm:.2f}
                </div>
                <div class="strat-profit" style="color:#4ade80;">
                    💰 Net Credit: ₹{net:,.2f}/lot &nbsp;(₹{net*lot_size:,.0f} total)
                </div>
                <div style="font-size:14px; color:#e2e8f0; margin-top:6px; font-weight:600;">
                    Breakevens: ₹{lower_be:,.0f} – ₹{upper_be:,.0f} &nbsp;⚠️ Unlimited risk
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif strat_choice == "Iron Butterfly":
            c_atm, p_atm = get_ltp(atm_strike)
            c_buy_ce, _ = get_ltp(ic_buy_call)
            _, p_buy_pe = get_ltp(ic_buy_put)
            net = max((c_atm + p_atm) - (c_buy_ce + p_buy_pe), 0)
            legs = [
                {"strike":ic_buy_put,"type":"PE","action":"BUY","premium":p_buy_pe},
                {"strike":atm_strike,"type":"PE","action":"SELL","premium":p_atm},
                {"strike":atm_strike,"type":"CE","action":"SELL","premium":c_atm},
                {"strike":ic_buy_call,"type":"CE","action":"BUY","premium":c_buy_ce},
            ]
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-title" style="color:#8b5cf6;">🦋 Iron Butterfly</div>
                <div class="strat-leg">
                    BUY 1× <b>{ic_buy_put} PE</b> @ ₹{p_buy_pe:.2f}<br>
                    SELL 1× <b>{atm_strike} PE</b> @ ₹{p_atm:.2f}<br>
                    SELL 1× <b>{atm_strike} CE</b> @ ₹{c_atm:.2f}<br>
                    BUY 1× <b>{ic_buy_call} CE</b> @ ₹{c_buy_ce:.2f}
                </div>
                <div class="strat-profit" style="color:#4ade80;">
                    💰 Net Credit: ₹{net:,.2f}/lot &nbsp;(₹{net*lot_size:,.0f} total)
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif strat_choice == "Bull Put Spread":
            sell_strike = ic_sell_put
            buy_strike = ic_buy_put
            _, p_sell = get_ltp(sell_strike)
            _, p_buy = get_ltp(buy_strike)
            net = max(p_sell - p_buy, 0)
            max_risk = diff - net
            legs = [
                {"strike":buy_strike,"type":"PE","action":"BUY","premium":p_buy},
                {"strike":sell_strike,"type":"PE","action":"SELL","premium":p_sell},
            ]
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-title" style="color:#22c55e;">📈 Bull Put Spread</div>
                <div class="strat-leg">
                    SELL 1× <b>{sell_strike} PE</b> @ ₹{p_sell:.2f}<br>
                    BUY 1× <b>{buy_strike} PE</b> @ ₹{p_buy:.2f}
                </div>
                <div class="strat-profit" style="color:#4ade80;">
                    💰 Net Credit: ₹{net:,.2f}/lot &nbsp;(₹{net*lot_size:,.0f} total)
                </div>
                <div style="color:#fca5a5; font-size:14px; margin-top:6px; font-weight:700;">
                    ⚠️ Max Risk: ₹{max_risk:,.2f}/lot
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif strat_choice == "Bear Call Spread":
            sell_strike = ic_sell_call
            buy_strike = ic_buy_call
            c_sell, _ = get_ltp(sell_strike)
            c_buy, _ = get_ltp(buy_strike)
            net = max(c_sell - c_buy, 0)
            max_risk = diff - net
            legs = [
                {"strike":sell_strike,"type":"CE","action":"SELL","premium":c_sell},
                {"strike":buy_strike,"type":"CE","action":"BUY","premium":c_buy},
            ]
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-title" style="color:#ef4444;">📉 Bear Call Spread</div>
                <div class="strat-leg">
                    SELL 1× <b>{sell_strike} CE</b> @ ₹{c_sell:.2f}<br>
                    BUY 1× <b>{buy_strike} CE</b> @ ₹{c_buy:.2f}
                </div>
                <div class="strat-profit" style="color:#4ade80;">
                    💰 Net Credit: ₹{net:,.2f}/lot &nbsp;(₹{net*lot_size:,.0f} total)
                </div>
                <div style="color:#fca5a5; font-size:14px; margin-top:6px; font-weight:700;">
                    ⚠️ Max Risk: ₹{max_risk:,.2f}/lot
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── P&L Heatmap ──
        if legs:
            st.markdown("#### 🗺️ P&L Heatmap (Spot × Days to Expiry)")
            spot_arr, dte_arr, pnl_mat = compute_pnl_heatmap(legs, lot_size, spot_price, diff)

            # Custom red-white-green colorscale
            colorscale = [
                [0, "#dc2626"],
                [0.35, "#fca5a5"],
                [0.5, "#1e293b"],
                [0.65, "#86efac"],
                [1, "#16a34a"],
            ]

            fig_hm = go.Figure(data=go.Heatmap(
                z=pnl_mat,
                x=np.round(spot_arr, 0),
                y=[f"DTE {int(d)}" for d in dte_arr],
                colorscale=colorscale,
                zmid=0,
                text=np.round(pnl_mat, 0).astype(int),
                texttemplate="%{text}",
                textfont=dict(size=9),
                hovertemplate="Spot: %{x}<br>%{y}<br>P&L: ₹%{z:,.0f}<extra></extra>",
                colorbar=dict(title="P&L (₹)", tickformat=","),
            ))
            fig_hm.update_layout(**PLOTLY_LAYOUT, height=350,
                title=dict(text=f"{strat_choice} P&L — {lot_size} lot", font=dict(size=13)),
                xaxis_title="Spot Price", yaxis_title="")
            fig_hm.add_vline(x=spot_price, line=dict(color="#f1f5f9", width=1.5, dash="dash"),
                             annotation_text="Current", annotation_position="top")
            st.plotly_chart(fig_hm, use_container_width=True)

            # Payoff at expiry line chart
            st.markdown("#### 📐 Payoff at Expiry")
            expiry_pnl = pnl_mat[0]  # DTE 0 row
            fig_payoff = go.Figure()
            # Color fill: green above 0, red below
            fig_payoff.add_trace(go.Scatter(
                x=spot_arr, y=expiry_pnl, mode="lines",
                line=dict(color="#e2e8f0", width=2), name="P&L",
                fill="tozeroy",
                fillcolor="rgba(34,197,94,0.15)",
            ))
            # Overlay red below zero
            neg_pnl = np.where(expiry_pnl < 0, expiry_pnl, 0)
            fig_payoff.add_trace(go.Scatter(
                x=spot_arr, y=neg_pnl, mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
                showlegend=False,
            ))
            fig_payoff.add_hline(y=0, line=dict(color="#475569", width=1))
            fig_payoff.add_vline(x=spot_price, line=dict(color="#3b82f6", dash="dash", width=1),
                                 annotation_text="Spot")
            fig_payoff.update_layout(**PLOTLY_LAYOUT, height=280,
                title=dict(text="Expiry Payoff Profile", font=dict(size=13)),
                xaxis_title="Spot at Expiry", yaxis_title="P&L (₹)")
            st.plotly_chart(fig_payoff, use_container_width=True)

    # ──────────────────────────────────
    #  TAB 4: CHARTS
    # ──────────────────────────────────
    with tab_charts:

        # OI Distribution
        st.markdown("#### 📊 Open Interest Distribution")
        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(
            x=df["Strike"], y=df["CE OI"], name="Call OI",
            marker=dict(color="#ef4444", opacity=0.85, line=dict(color="#b91c1c", width=0.5)),
            offsetgroup=0,
        ))
        fig_oi.add_trace(go.Bar(
            x=df["Strike"], y=df["PE OI"], name="Put OI",
            marker=dict(color="#22c55e", opacity=0.85, line=dict(color="#15803d", width=0.5)),
            offsetgroup=1,
        ))
        fig_oi.add_vline(x=atm_strike, line=dict(color="#f1f5f9", dash="dash", width=1), annotation_text="ATM")
        fig_oi.add_vline(x=max_pain, line=dict(color="#8b5cf6", dash="dashdot", width=1), annotation_text="Max Pain")
        fig_oi.update_layout(**PLOTLY_LAYOUT, height=380, barmode="group",
            title=dict(text=f"OI Distribution — {selected_index_name} ({selected_expiry})", font=dict(size=13)),
            xaxis_title="Strike", yaxis_title="Open Interest")
        st.plotly_chart(fig_oi, use_container_width=True)

        # OI Change chart
        st.markdown("#### 🔄 OI Change (Build-up / Unwinding)")
        fig_oichg = make_subplots(rows=1, cols=1)
        ce_chg_colors = ["#ef4444" if v >= 0 else "#7f1d1d" for v in df["CE OI Chg"]]
        pe_chg_colors = ["#22c55e" if v >= 0 else "#14532d" for v in df["PE OI Chg"]]
        fig_oichg.add_trace(go.Bar(x=df["Strike"]-diff*0.15, y=df["CE OI Chg"], name="CE OI Δ",
            marker_color=ce_chg_colors, width=diff*0.3))
        fig_oichg.add_trace(go.Bar(x=df["Strike"]+diff*0.15, y=df["PE OI Chg"], name="PE OI Δ",
            marker_color=pe_chg_colors, width=diff*0.3))
        fig_oichg.add_hline(y=0, line=dict(color="#475569", width=1))
        fig_oichg.update_layout(**PLOTLY_LAYOUT, height=320,
            title=dict(text="OI Change — Positive = Build-up, Negative = Unwinding", font=dict(size=13)),
            xaxis_title="Strike", yaxis_title="OI Change")
        st.plotly_chart(fig_oichg, use_container_width=True)

        ch1, ch2 = st.columns(2)

        with ch1:
            # Delta Skew
            st.markdown("#### 📉 Delta Skew")
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(x=df["Strike"], y=df["CE Delta"], mode="lines+markers",
                line=dict(color="#3b82f6", width=2), marker=dict(size=4), name="CE Δ"))
            fig_d.add_trace(go.Scatter(x=df["Strike"], y=df["PE Delta"], mode="lines+markers",
                line=dict(color="#ef4444", width=2), marker=dict(size=4), name="PE Δ"))
            fig_d.add_hline(y=0, line=dict(color="#475569", width=1))
            fig_d.add_vline(x=atm_strike, line=dict(color="#64748b", dash="dash", width=1))
            fig_d.update_layout(**PLOTLY_LAYOUT, height=320,
                title=dict(text="Delta Across Strikes", font=dict(size=12)),
                xaxis_title="Strike", yaxis_title="Delta")
            st.plotly_chart(fig_d, use_container_width=True)

        with ch2:
            # IV Smile
            st.markdown("#### 😊 IV Smile")
            fig_iv = go.Figure()
            fig_iv.add_trace(go.Scatter(x=df["Strike"], y=df["CE IV"], mode="lines+markers",
                line=dict(color="#3b82f6", width=2), marker=dict(size=4, symbol="triangle-up"), name="CE IV"))
            fig_iv.add_trace(go.Scatter(x=df["Strike"], y=df["PE IV"], mode="lines+markers",
                line=dict(color="#ef4444", width=2), marker=dict(size=4, symbol="triangle-down"), name="PE IV"))
            fig_iv.add_vline(x=atm_strike, line=dict(color="#64748b", dash="dash", width=1))
            fig_iv.update_layout(**PLOTLY_LAYOUT, height=320,
                title=dict(text="IV Smile / Skew", font=dict(size=12)),
                xaxis_title="Strike", yaxis_title="IV (%)")
            st.plotly_chart(fig_iv, use_container_width=True)

        # Cumulative OI Pressure
        st.markdown("#### ⚖️ Cumulative OI Pressure")
        ds = df.sort_values("Strike")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=ds["Strike"], y=ds["CE OI"].cumsum(), mode="lines",
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
            line=dict(color="#ef4444", width=2), name="Cumul. Call OI"))
        fig_cum.add_trace(go.Scatter(x=ds["Strike"], y=ds["PE OI"].cumsum(), mode="lines",
            fill="tozeroy", fillcolor="rgba(34,197,94,0.12)",
            line=dict(color="#22c55e", width=2), name="Cumul. Put OI"))
        fig_cum.add_vline(x=atm_strike, line=dict(color="#64748b", dash="dash", width=1))
        fig_cum.update_layout(**PLOTLY_LAYOUT, height=320,
            title=dict(text="Cumulative OI Build-up", font=dict(size=13)),
            xaxis_title="Strike", yaxis_title="Cumulative OI")
        st.plotly_chart(fig_cum, use_container_width=True)

        # Volume-Weighted Average Strike
        total_vol = df["CE Vol"].sum() + df["PE Vol"].sum()
        if total_vol > 0:
            vwas = ((df["Strike"] * (df["CE Vol"] + df["PE Vol"])).sum()) / total_vol
            st.markdown(f"""
            <div style="text-align:center; padding:10px; margin:8px 0;
                        border:1px solid #1e293b; border-radius:8px; background:rgba(17,24,39,0.5);">
                <span style="font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px;">
                    Volume-Weighted Avg Strike
                </span><br>
                <span style="font-family:'JetBrains Mono',monospace; font-size:24px; font-weight:700; color:#f59e0b;">
                    {vwas:,.1f}
                </span>
                <span style="font-size:12px; color:#64748b;"> &nbsp;(where the money flows)</span>
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    #  FOOTER
    # ═══════════════════════════════════════════════

    st.markdown("---")
    f1, f2, f3 = st.columns(3)
    f1.caption(f"🕐 {datetime.now().strftime('%H:%M:%S IST')}")
    f2.caption(f"📅 Expiry: {selected_expiry} · DTE: {tte_days:.1f}d")
    f3.caption(f"📊 {len(df)} strikes loaded · Lot: {lot_size}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

except ValueError as ve:
    st.error(f"🚫 **Data Error**: {ve}")
except requests.exceptions.HTTPError as he:
    code = he.response.status_code if he.response is not None else "?"
    st.error(f"🚫 **HTTP {code}**: Token likely expired. Re-authenticate via Upstox OAuth.")
except requests.exceptions.ConnectionError:
    st.error("🚫 **Connection Error**: Can't reach Upstox API.")
except requests.exceptions.Timeout:
    st.error("🚫 **Timeout**: API didn't respond in time.")
except Exception as e:
    st.error(f"🚫 {type(e).__name__}: {e}")
    st.exception(e)
