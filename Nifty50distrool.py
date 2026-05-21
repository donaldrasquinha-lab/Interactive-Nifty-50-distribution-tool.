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
    "NIFTY 50": {"key": "NSE_INDEX|Nifty 50", "symbol": "NIFTY", "diff": 50},
    "BANK NIFTY": {"key": "NSE_INDEX|Nifty Bank", "symbol": "BANKNIFTY", "diff": 100},
    "FINNIFTY": {"key": "NSE_INDEX|Nifty Fin Service", "symbol": "FINNIFTY", "diff": 50},
    "MIDCAP NIFTY": {"key": "NSE_INDEX|NIFTY MID SELECT", "symbol": "MIDCPNIFTY", "diff": 25},
}

# ── Upstox API Helper ──
class UpstoxClient:
    BASE = "https://api.upstox.com/v2"

    def __init__(self, token: str):
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def get_spot_price(self, instrument_key: str):
        url = f"{self.BASE}/market-quote/quotes"
        r = requests.get(url, headers=self.headers, params={"instrument_key": instrument_key}, timeout=10)
        r.raise_for_status()
        data = r.json()
        quote_key = list(data.get("data", {}).keys())[0]
        return data["data"][quote_key]["last_price"]

    def get_expiries(self, instrument_key: str):
        url = f"{self.BASE}/option/contract"
        r = requests.get(url, headers=self.headers, params={"instrument_key": instrument_key}, timeout=10)
        r.raise_for_status()
        data = r.json()
        expiries = sorted(set(
            c.get("expiry", "")[:10] if isinstance(c.get("expiry"), str) else str(c.get("expiry", ""))[:10]
            for c in data.get("data", [])
        ))
        return [e for e in expiries if e and e != "None"]

    def get_option_chain(self, instrument_key: str, expiry_date: str):
        url = f"{self.BASE}/option/chain"
        params = {"instrument_key": instrument_key, "expiry_date": expiry_date}
        r = requests.get(url, headers=self.headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])

    def get_historical_candles(self, instrument_key: str, interval: str = "day", days: int = 30):
        """Fetch intraday or daily OHLC candles for ADX computation."""
        from datetime import timedelta
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"{self.BASE}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
        r = requests.get(url, headers=self.headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        candles = data.get("data", {}).get("candles", [])
        # candles: [[timestamp, open, high, low, close, volume, oi], ...]
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
    """
    Compute ADX, +DI, -DI from OHLC candle data.
    Uses Wilder's smoothing method.
    Returns dict with adx, plus_di, minus_di (latest values).
    """
    if candles_df.empty or len(candles_df) < period + 2:
        return None

    df = candles_df.copy()
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_close"] = df["close"].shift(1)

    # True Range
    df["tr"] = df.apply(lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1)

    # Directional Movement
    df["+dm"] = df.apply(lambda r: max(r["high"] - r["prev_high"], 0)
                         if pd.notna(r["prev_high"]) and (r["high"] - r["prev_high"]) > (r["prev_low"] - r["low"])
                         else 0, axis=1)
    df["-dm"] = df.apply(lambda r: max(r["prev_low"] - r["low"], 0)
                         if pd.notna(r["prev_low"]) and (r["prev_low"] - r["low"]) > (r["high"] - r["prev_high"])
                         else 0, axis=1)

    df = df.iloc[1:].reset_index(drop=True)  # drop first row (NaN prev)

    # Wilder's smoothing
    tr_smooth = [df["tr"].iloc[:period].sum()]
    pdm_smooth = [df["+dm"].iloc[:period].sum()]
    ndm_smooth = [df["-dm"].iloc[:period].sum()]

    for i in range(period, len(df)):
        tr_smooth.append(tr_smooth[-1] - (tr_smooth[-1] / period) + df["tr"].iloc[i])
        pdm_smooth.append(pdm_smooth[-1] - (pdm_smooth[-1] / period) + df["+dm"].iloc[i])
        ndm_smooth.append(ndm_smooth[-1] - (ndm_smooth[-1] / period) + df["-dm"].iloc[i])

    # +DI / -DI series
    plus_di_list = []
    minus_di_list = []
    dx_list = []

    for i in range(len(tr_smooth)):
        tr_val = tr_smooth[i]
        pdi = (pdm_smooth[i] / tr_val * 100) if tr_val > 0 else 0
        ndi = (ndm_smooth[i] / tr_val * 100) if tr_val > 0 else 0
        plus_di_list.append(pdi)
        minus_di_list.append(ndi)
        denom = pdi + ndi
        dx_list.append(abs(pdi - ndi) / denom * 100 if denom > 0 else 0)

    if len(dx_list) < period:
        return None

    # ADX = Wilder's smooth of DX
    adx_list = [sum(dx_list[:period]) / period]
    for i in range(period, len(dx_list)):
        adx_list.append((adx_list[-1] * (period - 1) + dx_list[i]) / period)

    return {
        "adx": round(adx_list[-1], 2),
        "plus_di": round(plus_di_list[-1], 2),
        "minus_di": round(minus_di_list[-1], 2),
    }


# ── Sidebar Setup ──
st.sidebar.markdown('<div class="sidebar-header">UPSTOX Engine</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-title">OI Analyzer Pro</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

api_token = st.sidebar.text_input("Upstox Access Token (Bearer)", type="password", help="Paste only the raw alphanumeric token here. Do NOT type 'Bearer'.")
selected_idx_label = st.sidebar.selectbox("Select Target Asset Index:", list(INDICES.keys()))

lot_size = INDICES[selected_idx_label]["lot_size"]
instrument_key = INDICES[selected_idx_label]["key"]
oi_step = INDICES[selected_idx_label]["diff"]

client = UpstoxClient(api_token if api_token else "SIMULATED_TOKEN")

spot_price = INDICES[selected_idx_label]["default_spot"]
is_live = False
raw_data = []

# Fetch Expiries Array
expiries_list = client.get_expiries(instrument_key)
selected_expiry = st.sidebar.selectbox("Select Expiry Date:", expiries_list)

# Volatility and Parameters Configuration Widgets
st.sidebar.header("🔧 Alpha System Multipliers")
iv_percent = st.sidebar.slider("Implied Volatility (IV %)", 5.0, 40.0, 12.0, 0.5) / 100
today_cal = datetime.now()
days_until_thursday_cal = (3 - today_cal.weekday()) % 7
if days_until_thursday_cal == 0 and today_cal.hour >= 16:
    days_until_thursday_cal = 7
days_to_expiry = st.sidebar.number_input("Days to Expiry (DTE Scalar)", 1, 30, int(max(1, days_until_thursday_cal)))
show_adjustment = st.sidebar.checkbox("Overlay Recommended Adjustment Leg", value=True)

st.sidebar.header("⏱️ Live Automations")
auto_refresh = st.sidebar.checkbox("Enable Real-Time Streaming (Auto Rerun)", value=False)
refresh_interval = st.sidebar.slider("Refresh Data Every (Seconds)", min_value=2, max_value=30, value=5)

# ── 3. Data Processing Matrix Ingestion Layer ──
time_factor = np.sqrt(days_to_expiry / 365)
expected_move = spot_price * iv_percent * time_factor
atm_premium = int(max(25.0, round(expected_move * 0.4)))
oi_wall_premium = int(max(10.0, round(atm_premium * 0.45)))
hedge_premium = int(max(2.0, round(oi_wall_premium * 0.35)))

atm_strike = int(round(spot_price / oi_step) * oi_step)
oi_wall_strike = atm_strike + oi_step

live_pcr = 0.95
oi_support = atm_strike - oi_step
oi_resistance = atm_strike + oi_step
money_velocity_ratio = 1.05  
volatility_skew_index = 1.02 

# ── Live Data Extract & Validation Engine ──
if api_token:
    try:
        quote_response = client.get_spot_price(instrument_key)
        
        if quote_response.status_code == 200 and 'application/json' in quote_response.headers.get('Content-Type', ''):
            quote_res = quote_response.json()
            if quote_res.get('status') == 'success':
                spot_price = quote_res['data'][instrument_key]['last_price']
                atm_strike = int(round(spot_price / oi_step) * oi_step)
                
                chain_response = client.get_option_chain(instrument_key, selected_expiry)
                
                if chain_response.status_code == 200 and 'application/json' in chain_response.headers.get('Content-Type', ''):
                    chain_res = chain_response.json()
                    if chain_res.get('status') == 'success' and len(chain_res.get('data', [])) > 0:
                        raw_data = chain_res['data']
                        
                        max_call_oi, max_put_oi = -1, -1
                        total_call_oi, total_put_oi = 0, 0
                        total_call_coi, total_put_coi = 0, 0
                        total_call_iv, total_put_iv = 0.0, 0.0
                        
                        best_call_strike, best_put_strike = atm_strike + oi_step, atm_strike - oi_step
                        premium_lookup = {}
                        processed_records = []
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        for item in raw_data:
                            strike = int(item['strike_price'])
                            ce_data = item.get('call_options', {}).get('market_data', {}) if item.get('call_options') else {}
                            pe_data = item.get('put_options', {}).get('market_data', {}) if item.get('put_options') else {}
                            
                            c_oi, p_oi = ce_data.get('oi', 0), pe_data.get('oi', 0)
                            c_coi, p_coi = ce_data.get('oi_change', 0), pe_data.get('oi_change', 0)
                            c_iv, p_iv = ce_data.get('implied_volatility', 12.0), pe_data.get('implied_volatility', 12.0)
                            
                            total_call_oi += c_oi
                            total_put_oi += p_oi
                            total_call_coi += abs(c_coi)
                            total_put_coi += abs(p_coi)
                            
                            if abs(strike - atm_strike) <= (oi_step * 3):
                                total_call_iv += c_iv
                                total_put_iv += p_iv
                            
                            premium_lookup[strike] = ce_data.get('ltp', atm_premium)
                            
                            if strike > spot_price and c_oi > max_call_oi:
                                max_call_oi = c_oi
                                best_call_strike = strike
                            if strike < spot_price and p_oi > max_put_oi:
                                max_put_oi = p_oi
                                best_put_strike = strike
                                
                            processed_records.append({
                                "Timestamp": timestamp_str, "Underlying": selected_idx_label, "Spot_Price": spot_price,
                                "Expiry_Date": selected_expiry, "Strike_Price": strike, "CE_LTP": ce_data.get('ltp', 0.0),
                                "CE_OI": c_oi, "PE_LTP": pe_data.get('ltp', 0.0), "PE_OI": p_oi
                            })
                        
                        oi_wall_strike = best_call_strike
                        oi_resistance, oi_support = best_call_strike, best_put_strike
                        live_pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 1.0
                        
                        money_velocity_ratio = round(total_put_coi / total_call_coi, 2) if total_call_coi > 0 else 1.0
                        volatility_skew_index = round(total_put_iv / total_call_iv, 2) if total_call_iv > 0 else 1.0
                        is_live = True
                        
                        atm_premium = premium_lookup.get(atm_strike, atm_premium)
                        oi_wall_premium = premium_lookup.get(oi_wall_strike, oi_wall_premium)
                        
                        strike_sell = oi_wall_strike
                        strike_hedge = strike_sell + (strike_sell - atm_strike)
                        hedge_premium = premium_lookup.get(strike_hedge, hedge_premium)
                        
                        # Store structural records onto server disk arrays
                        filename_prefix = selected_idx_label.replace(" ", "_").lower()
                        pd.DataFrame(processed_records).to_csv(f"{filename_prefix}_chain_latest.csv", index=False)
                        st.sidebar.success("🟢 Connected & Syncing Live!")
                    else:
                        st.sidebar.error("❌ Expiry Mismatch: Upstox option chain empty for this date string.")
                else:
                    st.sidebar.error(f"❌ Option Chain Refused. HTTP: {chain_response.status_code}")
            else:
                st.sidebar.error("❌ Token Denied: Upstox rejected developer app verification payload.")
        else:
            st.sidebar.error(f"❌ LTP Fetch Blocked. HTTP: {quote_response.status_code}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Diagnostic Warning: {str(e)}")

if not is_live:
    atm_strike = int(round(spot_price / oi_step) * oi_step)
    oi_resistance = atm_strike + oi_step
    oi_support = atm_strike - oi_step
    oi_wall_strike = oi_resistance
    live_pcr = 0.95

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
    st.metric(label=f"🎯 {selected_idx_label} Spot", value=f"₹{spot_price:,.2f}", delta="Live Feed Active" if is_live else "Simulated Engine")
with col_m2:
    st.metric(label="📊 Put-Call Ratio (PCR)", value=f"{live_pcr}", delta=f"Velocity Ratio: {money_velocity_ratio}")
with col_m3:
    st.metric(label="🟢 Live Volatility Skew", value=f"{volatility_skew_index}x", delta="Skew Anomalous" if volatility_skew_index > 1.05 else "Normal Curve")
with col_m4:
    st.metric(label="🔴 Trend Guard Status", value="EXPLOSIVE BREAKOUT" if is_explosive_trend else "CONSOLIDATING Range", delta=f"Decay Scalar: {round(decay_efficiency_factor, 2)}x")
with col_m5:
    st.metric(label="📅 Active Options Expiry", value=selected_expiry)

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
    ax_p.fill_between(x, prob_density, 0, where=(x >= sd1_lower) & (x <= sd1_upper), facecolor='#10b981', alpha=0.2, label='68.2% (1 SD)')
    ax_p.fill_between(x, prob_density, 0, where=((x >= sd2_lower) & (x < sd1_lower)) | ((x > sd1_upper) & (x <= sd2_upper)), facecolor='#f59e0b', alpha=0.12, label='95.4% (2 SD)')
    ax_p.axvline(spot_price, color='#4f46e5', linestyle=':', linewidth=1.5, label='Spot Line')
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

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
