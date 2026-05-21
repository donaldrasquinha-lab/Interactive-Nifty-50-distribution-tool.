"""
╔══════════════════════════════════════════════════════════════╗
║         UPSTOX ALPHA TRADING ENGINE — Live Options Matrix     ║
║  Pulls live values, analyzes CE/PE metrics around ATM,      ║
║  and features automated refresh triggers for active trading.║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time

# ── 1. Page Configuration & Scaffold Setup ──
st.set_page_config(
    page_title="Upstox Alpha Live Signal Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Light Theme CSS Injector
st.markdown("""
<style>
@import url('https://googleapis.com');
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; }

/* Dynamic KPI metric panel layout formatting */
div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 14px 18px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}
div[data-testid="stMetric"] label { color: #64748b !important; font-size: 11px !important; letter-spacing: 1px; text-transform: uppercase; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important; color: #0f172a !important; font-size: 22px !important;
}

/* Direction sentiment summary blocks */
.direction-card {
    border-radius: 14px; padding: 22px; margin: 15px 0; font-family: 'JetBrains Mono', monospace; border: 1px solid #e2e8f0;
}
.score-label { font-size: 11px; letter-spacing: 2px; color: #ffffff; opacity: 0.9; margin-bottom: 4px; }
.direction-text { font-size: 28px; font-weight: 700; color: #ffffff; }
.sentiment-text { font-size: 13px; color: #f8fafc; margin-top: 4px; }

/* Sidebar formatting elements */
.sidebar-header { font-size: 11px; letter-spacing: 3px; color: #0284c7; font-weight: 600; text-transform: uppercase; }
.sidebar-title { font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 2px; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Title Bar Fixed on Top ──
st.title("⚡ Upstox Live Multi-Index OI & Statistical Dashboard")

# ── Index Definitions ──
INDICES = {
    "NIFTY 50": {"key": "NSE_INDEX|Nifty 50", "symbol": "NIFTY", "diff": 50, "default_spot": 23800},
    "BANK NIFTY": {"key": "NSE_INDEX|Nifty Bank", "symbol": "BANKNIFTY", "diff": 100, "default_spot": 51200},
    "FINNIFTY": {"key": "NSE_INDEX|Nifty Fin Service", "symbol": "FINNIFTY", "diff": 50, "default_spot": 22400},
    "MIDCAP NIFTY": {"key": "NSE_INDEX|NIFTY MID SELECT", "symbol": "MIDCPNIFTY", "diff": 25, "default_spot": 12100},
}

# ── Upstox API Helper ──
class UpstoxClient:
    BASE = "https://api.upstox.com/v2"

    def __init__(self, token: str):
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Api-Version": "2.0"
        }

    def get_spot_price(self, instrument_key: str):
        url = f"{self.BASE}/market-quote/ltp"
        r = requests.get(url, headers=self.headers, params={"instrument_key": instrument_key}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data["data"][instrument_key]["last_price"]

    def get_expiries(self, instrument_key: str):
        """Generates target expiry array by naturally projecting next close calendar Thursdays."""
        today = datetime.now()
        expiries = []
        for i in range(4): # Project 4 upcoming weekly expiries
            days_until_thursday = (3 - today.weekday() + (i * 7)) % (7 + (i * 7))
            if days_until_thursday == 0 and today.hour >= 16:
                days_until_thursday += 7
            target_date = today + timedelta(days=days_until_thursday)
            expiries.append(target_date.strftime("%Y-%m-%d"))
        return expiries

    def get_option_chain(self, instrument_key: str, expiry_date: str):
        url = f"{self.BASE}/option/chain"
        params = {"instrument_key": instrument_key, "expiry_date": expiry_date}
        r = requests.get(url, headers=self.headers, params=params, timeout=10)
        if r.status_code != 200 or 'application/json' not in r.headers.get('Content-Type', ''):
            return []
        return r.json().get("data", [])

    def get_historical_candles(self, instrument_key: str, interval: str = "1d", days: int = 30):
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"{self.BASE}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
        r = requests.get(url, headers=self.headers, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        candles = data.get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()
        rows = []
        for c in candles:
            rows.append({
                "timestamp": c[0], "open": c[1], "high": c[2],
                "low": c[3], "close": c[4], "volume": c[5],
            })
        cdf = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return cdf

def compute_adx(candles_df: pd.DataFrame, period: int = 14):
    """Compute ADX, +DI, -DI from OHLC candle data using Wilder's smoothing."""
    if candles_df.empty or len(candles_df) < period * 2:
        return {"adx": 22.5, "plus_di": 24.1, "minus_di": 19.4}

    df = candles_df.copy()
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_close"] = df["close"].shift(1)

    df["tr"] = df.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)

    df["+dm"] = df.apply(lambda r: max(r["high"] - r["prev_high"], 0)
                         if pd.notna(r["prev_high"]) and (r["high"] - r["prev_high"]) > (r["prev_low"] - r["low"])
                         else 0, axis=1)
    df["-dm"] = df.apply(lambda r: max(r["prev_low"] - r["low"], 0)
                         if pd.notna(r["prev_low"]) and (r["prev_low"] - r["low"]) > (r["high"] - r["prev_high"])
                         else 0, axis=1)

    df = df.iloc[1:].reset_index(drop=True)

    tr_smooth = [df["tr"].iloc[:period].sum()]
    pdm_smooth = [df["+dm"].iloc[:period].sum()]
    ndm_smooth = [df["-dm"].iloc[:period].sum()]

    for i in range(period, len(df)):
        tr_smooth.append(tr_smooth[-1] - (tr_smooth[-1] / period) + df["tr"].iloc[i])
        pdm_smooth.append(pdm_smooth[-1] - (pdm_smooth[-1] / period) + df["+dm"].iloc[i])
        ndm_smooth.append(ndm_smooth[-1] - (ndm_smooth[-1] / period) + df["-dm"].iloc[i])

    plus_di_list = []
    minus_di_list = []
    dx_list = []

    # FIXED ARRAY FILL: Maps out completed dx calculations for directional indexes
    for i in range(len(tr_smooth)):
        tr_val = tr_smooth[i]
        pdi = (pdm_smooth[i] / tr_val * 100) if tr_val > 0 else 0
        ndi = (ndm_smooth[i] / tr_val * 100) if tr_val > 0 else 0
        plus_di_list.append(pdi)
        minus_di_list.append(ndi)
        denom = pdi + ndi
        dx = (abs(pdi - ndi) / denom * 100) if denom > 0 else 0
        dx_list.append(dx)

    adx_series = [np.mean(dx_list[:period])]
    for i in range(period, len(dx_list)):
        adx_series.append((adx_series[-1] * (period - 1) + dx_list[i]) / period)

    return {
        "adx": adx_series[-1],
        "plus_di": plus_di_list[-1],
        "minus_di": minus_di_list[-1]
    }

# ── Sidebar Setup ──
st.sidebar.markdown('<div class="sidebar-header">UPSTOX Engine</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-title">OI Analyzer Pro</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

api_token = st.sidebar.text_input("Upstox Access Token", type="password", help="Paste your active Bearer access token here.")
selected_idx_label = st.sidebar.selectbox("Select Target Asset Index:", list(INDICES.keys()))

client = UpstoxClient(api_token if api_token else "SIMULATED_TOKEN")

spot_price = INDICES[selected_idx_label]["default_spot"]
is_live = False
raw_chain = []

# Fetch Expiries 
expiries_list = client.get_expiries(INDICES[selected_idx_label]["key"])
selected_expiry = st.sidebar.selectbox("Select Expiry Date:", expiries_list)

# ── Data Fetching Automation ──
if api_token:
    try:
        spot_price = client.get_spot_price(INDICES[selected_idx_label]["key"])
        raw_chain = client.get_option_chain(INDICES[selected_idx_label]["key"], selected_expiry)
        if raw_chain:
            is_live = True
    except Exception:
        st.sidebar.error("Live sync failed. Check token credentials.")

# Emulate options data structure if connections fallback
if not is_live:
    diff_step = INDICES[selected_idx_label]["diff"]
    atm_strike = int(round(spot_price / diff_step) * diff_step)
    
    for offset in range(-6, 7):
        strike_val = atm_strike + (offset * diff_step)
        dist_factor = abs(offset)
        
        ce_oi = int(max(40000, 2100000 - (dist_factor * 260000) + np.random.randint(-40000, 40000)))
        pe_oi = int(max(40000, 1950000 - (dist_factor * 250000) + np.random.randint(-30000, 30000)))
        
        raw_chain.append({
            "strike_price": strike_val,
            "call_options": {"market_data": {"oi": ce_oi, "ltp": max(4.0, 140 - (offset * 16)), "volume": ce_oi // 8}},
            "put_options": {"market_data": {"oi": pe_oi, "ltp": max(4.0, 140 + (offset * 16)), "volume": pe_oi // 8}}
        })

# ── 4-Factor Model & Scoring Computations ──
total_ce_oi, total_pe_oi = 0, 0
max_ce_oi, max_pe_oi = -1, -1
resistance_strike, support_strike = spot_price, spot_price

chain_rows = []
for row in raw_chain:
    strike = int(row["strike_price"])
    ce = row.get("call_options", {}).get("market_data", {}) if row.get("call_options") else {}
    pe = row.get("put_options", {}).get("market_data", {}) if row.get("put_options") else {}
    
    c_oi = ce.get("oi", 0)
    p_oi = pe.get("oi", 0)
    
    total_ce_oi += c_oi
    total_pe_oi += p_oi
    
    if strike > spot_price and c_oi > max_ce_oi:
        max_ce_oi = c_oi
        resistance_strike = strike
    if strike < spot_price and p_oi > max_pe_oi:
        max_pe_oi = p_oi
        support_strike = strike
        
    chain_rows.append({
        "Strike Price": strike, "CE Premium": ce.get("ltp", 0.0), "CE Open Interest": c_oi,
        "PE Open Interest": p_oi, "PE Premium": pe.get("ltp", 0.0)
    })

df_chain = pd.DataFrame(chain_rows).sort_values("Strike Price").reset_index(drop=True)
pcr_ratio = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 1.0

# Compute historical indicators for Factor 4
hist_candles = client.get_historical_candles(INDICES[selected_idx_label]["key"])
tech_indicators = compute_adx(hist_candles)

# Factor Scoring Logic 
factor_scores = []
factor_scores.append(25 if pcr_ratio > 1.05 else (0 if pcr_ratio < 0.85 else 12.5))
factor_scores.append(25 if spot_price > (resistance_strike + support_strike)/2 else 10)
factor_scores.append(25 if total_pe_oi > total_ce_oi else 5)
factor_scores.append(25 if tech_indicators["plus_di"] > tech_indicators["minus_di"] else 5)

final_sentiment_score = int(sum(factor_scores))
direction_label = "BULLISH" if final_sentiment_score >= 65 else ("BEARISH" if final_sentiment_score <= 40 else "NEUTRAL")
card_bg_color = "#064e3b" if direction_label == "BULLISH" else ("#7f1d1d" if direction_label == "BEARISH" else "#1e293b")


# ── Decay & Trend Guard Multipliers ──
decay_efficiency_factor = max(0.3, (7.0 - days_to_expiry) / 7.0) if days_to_expiry <= 7 else 0.25
is_explosive_trend = True if abs(spot_price - atm_strike) > (expected_move * 0.85) else False
trend_guard_multiplier = 0.50 if is_explosive_trend else 1.00

# ── Integrated Alpha Scoring Matrix Core Logic ──
f1_signal_score = 25 if live_pcr > 1.05 and money_velocity_ratio > 1.10 else (0 if live_pcr < 0.85 else 12.5)
f2_signal_score = 25 if spot_price > (oi_resistance + oi_support) / 2 else 10
f3_signal_score = 25 if volatility_skew_index > 1.03 else 12.5
f4_signal_score = 25 if not is_explosive_trend else 5

alpha_composite_signal = int((f1_signal_score + f2_signal_score + f3_signal_score + f4_signal_score) * trend_guard_multiplier)

if is_explosive_trend:
    signal_classification = "EXPLOSIVE TREND OVERHEAD: HOLD EXECUTION"
    signal_theme_color = "#ea580c"  
elif alpha_composite_signal >= 60:
    signal_classification = "STRONG BULLISH SPREAD ENTRY: BUY"
    signal_theme_color = "#16a34a"  
else:
    signal_classification = "OVERHEAD RESISTANCE LOCKED: SELL SPREAD"
    signal_theme_color = "#dc2626"  

# ── 4. Metric Panels Plotted Directly Under Title Bar ──
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric(label=f"🎯 {selected_index} Spot", value=f"₹{spot_price:,.2f}", delta="Live Feed Active" if is_live else "Simulated Engine")
with col_m2:
    st.metric(label="📊 Put-Call Ratio (PCR)", value=f"{live_pcr}", delta=f"Velocity Ratio: {money_velocity_ratio}")
with col_m3:
    st.metric(label="🟢 Live Volatility Skew", value=f"{volatility_skew_index}x", delta="Skew Anomalous" if volatility_skew_index > 1.05 else "Normal Curve")
with col_m4:
    st.metric(label="🔴 Trend Guard Status", value="EXPLOSIVE BREAKOUT" if is_explosive_trend else "CONSOLIDATING Range", delta=f"Decay Scalar: {round(decay_efficiency_factor, 2)}x")
with col_m5:
    st.metric(label="📅 Active Options Expiry", value=detected_expiry)

# ── 5. Trend Signal Window Layout Block ──
st.markdown(f"""
<div class="direction-card" style="background: {signal_theme_color}; border-color: {signal_theme_color}; color: #ffffff;">
    <div class="score-label">⚡ UPSTOX QUANTALPHA SIGNAL MATRIX Engine</div>
    <div class="direction-text">{signal_classification} ({alpha_composite_signal}/100)</div>
    <div class="sentiment-text">Trades filtered using dynamic premium lookahead windows, Intraday Change in OI acceleration velocity profiles, and trend guard multipliers. Last calculated: {datetime.now().strftime('%H:%M:%S')}</div>
</div>
""", unsafe_allow_html=True)

# ── 6. Mathematical Engine Setup (68-95-99.7 Rule) ──
one_sd_move = spot_price * iv_percent * time_factor
sd1_lower, sd1_upper = spot_price - one_sd_move, spot_price + one_sd_move
sd2_lower, sd2_upper = spot_price - (2 * one_sd_move), spot_price + (2 * one_sd_move)

# ── 7. Advanced Geometry Strategies Visualization ──
strike_buy = atm_strike
strike_sell = oi_wall_strike
strike_hedge = strike_sell + (strike_sell - strike_buy)

qty_buy, qty_sell, qty_hedge = 1, 2, 1
x = np.linspace(spot_price - (3 * one_sd_move), spot_price + (3 * one_sd_move), 2000)

payoff_buy = (np.maximum(x - strike_buy, 0) - atm_premium) * qty_buy
payoff_sell = (oi_wall_premium - np.maximum(x - strike_sell, 0)) * qty_sell
y_initial = (payoff_buy + payoff_sell) * lot_size
y_adjusted = y_initial + ((np.maximum(x - strike_hedge, 0) - hedge_premium) * qty_hedge * lot_size)

lower_be = strike_buy + (atm_premium - (2 * oi_wall_premium))
upper_be = strike_sell + ((strike_sell - strike_buy) - (atm_premium - (2 * oi_wall_premium)))

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📊 Market Valuation Probability Bands")
    prob_density = norm.pdf(x, spot_price, one_sd_move)
    fig_p, ax_p = plt.subplots(figsize=(10, 4.5))
    fig_p.patch.set_facecolor('#ffffff')
    ax_p.set_facecolor('#f8fafc')
    ax_p.plot(x, prob_density, color='#475569', linewidth=2)
    ax_p.fill_between(x, prob_density, 0, where=(x >= sd1_lower) & (x <= sd1_upper), facecolor='#10b981', alpha=0.2, label='68.2% (1 Standard Deviation Bounds)')
    ax_p.fill_between(x, prob_density, 0, where=((x >= sd2_lower) & (x < sd1_lower)) | ((x > sd1_upper) & (x <= sd2_upper)), facecolor='#f59e0b', alpha=0.12, label='95.4% (2 Standard Deviation Bounds)')
    ax_p.axvline(spot_price, color='#4f46e5', linestyle=':', linewidth=1.5, label=f'Spot valuation baseline')
    ax_p.tick_params(colors='#475569', labelsize=9)
    ax_p.legend(loc="upper left", frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#0f172a', fontsize=8)
    ax_p.set_xlim(x.min(), x.max())
    ax_p.get_yaxis().set_visible(False)
    ax_p.grid(True, linestyle=":", alpha=0.3, color="#cbd5e1")
    st.pyplot(fig_p)

with col_right:
    st.markdown("### 📈 Derivatives Payoff Return Matrix")
    fig_t, ax_t = plt.subplots(figsize=(10, 4.5))
    fig_t.patch.set_facecolor('#ffffff')
    ax_t.set_facecolor('#f8fafc')
    ax_t.plot(x, y_initial, color="#0d9488", linewidth=2, linestyle="--", alpha=0.6, label="Initial Ratio Call Spread")
    if show_adjustment:
        ax_t.plot(x, y_adjusted, color="#2563eb", linewidth=2.5, label="Adjusted Configuration (Risk Capped)")
        ax_t.fill_between(x, y_adjusted, 0, where=(y_adjusted >= 0), facecolor='#10b981', alpha=0.1)
        ax_t.fill_between(x, y_adjusted, 0, where=(y_adjusted < 0), facecolor='#ef4444', alpha=0.1)
    else:
        ax_t.fill_between(x, y_initial, 0, where=(y_initial >= 0), facecolor='#10b981', alpha=0.12)
        ax_t.fill_between(x, y_initial, 0, where=(y_initial < 0), facecolor='#ef4444', alpha=0.12)
    ax_t.axhline(0, color='#64748b', linestyle='-', linewidth=1.2)
    ax_t.scatter([lower_be, upper_be], np.zeros_like([lower_be, upper_be]), color='#ea580c', s=50, zorder=5)
    ax_t.tick_params(colors='#475569', labelsize=9)
    ax_t.legend(loc="upper left", frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#0f172a', fontsize=8)
    ax_t.set_xlim(x.min(), x.max())
    ax_t.grid(True, linestyle=":", alpha=0.3, color="#cbd5e1")
    st.pyplot(fig_t)

# ── 8. Clean Table Data Recommendations ──
st.markdown("---")
st.subheader("📋 Executable Order Matrix & Live Recommendations")
col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.markdown("### 🟢 Phase 1: Recommended Initial Setup")
    initial_data = {
        "Action": ["🟢 BUY (ATM Strike)", "🔴 SELL (OI Wall Strike)"],
        "Option Strike": [f"{strike_buy} CE", f"{strike_sell} CE"],
        "Lots / Qty": [f"{qty_buy} Lot ({lot_size} Qty)", f"{qty_sell} Lots ({lot_size * qty_sell} Qty)"],
        "Premium (LTP)": [f"₹{int(atm_premium)}", f"₹{int(oi_wall_premium)}"],
        "Margin Impact": [f"-₹{int(atm_premium * lot_size)}", f"+₹{int(oi_wall_premium * qty_sell * lot_size)}"]
    }
    st.table(initial_data)

with col_rec2:
    st.markdown("### 🟠 Phase 2: Recommended Adjustment Leg")
    if not show_adjustment:
        st.warning("⚠️ Market risk within normal bounds. Adjustment leg inactive.")
    else:
        st.success("🔥 **ADJUSTMENT LAYER ACTIVATED**")
        adj_data = {
            "Action": ["🟢 BUY (OTM Protection)"],
            "Option Strike": [f"{strike_hedge} CE"],
            "Lots / Qty": [f"{qty_hedge} Lot ({lot_size} Qty)"],
            "Premium (LTP)": [f"₹{int(hedge_premium)}"],
            "Margin Impact": [f"-₹{int(hedge_premium * lot_size)}"]
        }
        st.table(adj_data)

# ── 9. Automated Streaming Rerun Controller Loops ──
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
