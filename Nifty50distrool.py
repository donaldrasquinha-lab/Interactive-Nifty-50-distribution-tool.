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
# Corrected master instrument tokens matching Upstox asset dictionaries
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
        # Passing parameter exactly configured as 'instrument_key' query string 
        params = {"instrument_key": instrument_key}
        r = requests.get(url, headers=self.headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # Safe extraction guarding against malformed responses or errors
        if "data" in data and instrument_key in data["data"]:
            return float(data["data"][instrument_key]["last_price"])
        else:
            # Fallback check if the key returns encoded or formatted variations
            first_key = list(data.get("data", {}).keys())[0]
            return float(data["data"][first_key]["last_price"])

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
        """Fetch historical data. Fixed path sequencing format layout matching Upstox V2 specs."""
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Upstox V2 path template layout structure requirement: /historical-candle/{instrumentKey}/{interval}/{to_date}/{from_date}
        url = f"{self.BASE}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
        r = requests.get(url, headers=self.headers, timeout=10)
        r.raise_for_status()
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
    """Compute ADX, +DI, -DI from OHLC candle data using Wilder's smoothing method."""
    if candles_df.empty or len(candles_df) < period + 2:
        return None

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

    adx_list = [sum(dx_list[:period]) / period]
    for i in range(period, len(dx_list)):
        adx_list.append((adx_list[-1] * (period - 1) + dx_list[i]) / period)

    return {
        "adx": round(adx_list[-1], 2),
        "plus_di": round(plus_di_list[-1], 2),
        "minus_di": round(minus_di_list[-1], 2),
    }


def black_scholes_greeks(S, K, T, r, sigma, option_type="CE"):
    """Calculate option price, delta, gamma, theta, and vega using Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return {"price": 0, "delta": 0, "gamma": 0, "theta": 0, "vega": 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100 
    
    if option_type == "CE":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
    return {"price": round(price, 2), "delta": round(delta, 3), "gamma": round(gamma, 5), "theta": round(theta, 2), "vega": round(vega, 2)}


# ── Sidebar Setup ──
api_token = st.sidebar.text_input("🔑 Upstox Access Token", type="password", value="")
selected_index_name = st.sidebar.selectbox("🎯 Select Underlying Index", list(INDICES.keys()))

st.sidebar.markdown("---")
st.sidebar.header("🔧 Alpha System Multipliers")
iv_override = st.sidebar.slider("Flat Implied Volatility (%)", min_value=5.0, max_value=80.0, value=15.0, step=0.5) / 100
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=12.0, value=7.0, step=0.1) / 100
strike_depth = st.sidebar.slider("Strike Range Bracket Around ATM", min_value=3, max_value=15, value=7)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Enable Auto-Refresh (10s Triggers)", value=False)

# ── Core Engine Evaluation Loop ──
if not api_token:
    st.info("💡 Please input your Upstox API Bearer Token in the sidebar console to initialize the engine pipelines.")
else:
    try:
        client = UpstoxClient(token=api_token)
        index_meta = INDICES[selected_index_name]
        
        # Pull underlying execution matrix values
        spot_price = client.get_spot_price(index_meta["key"])
        expiries = client.get_expiries(index_meta["key"])
        
        if not expiries:
            st.error("No valid active derivative contract windows resolved from the broker API.")
            st.stop()
            
        selected_expiry = st.sidebar.selectbox("📅 Expiry Window Target", expiries, index=0)
        
        # Calculate dynamic time vectors
        expiry_dt = datetime.strptime(selected_expiry, "%Y-%m-%d").replace(hour=15, minute=30)
        time_to_expiry_days = (expiry_dt - datetime.now()).total_seconds() / (86400 * 365)
        time_to_expiry_days = max(time_to_expiry_days, 0.0001) 
        
        # Fetch technical structure indicators
        with st.spinner("Analyzing Index Volatility Structure..."):
            candles_df = client.get_historical_candles(index_meta["key"], interval="day", days=30)
            adx_metrics = compute_adx(candles_df)
            chain_raw = client.get_option_chain(index_meta["key"], selected_expiry)
            
        if not chain_raw:
            st.warning("Empty execution array returned for the target strike range matrix configuration.")
            st.stop()

        # Compute dynamic rounded boundary levels 
        diff = index_meta["diff"]
        atm_strike = round(spot_price / diff) * diff
        
        # Process and build option chain table
        chain_records = []
        for strike_data in chain_raw:
            strike_price = float(strike_data.get("strike_price", 0))
            if abs(strike_price - atm_strike) <= (strike_depth * diff):
                ce = strike_data.get("call_options", {})
                pe = strike_data.get("put_options", {})
                
                ce_oi = ce.get("market_data", {}).get("oi", 0) if ce else 0
                pe_oi = pe.get("market_data", {}).get("oi", 0) if pe else 0
                ce_ltp = ce.get("market_data", {}).get("ltp", 0) if ce else 0
                pe_ltp = pe.get("market_data", {}).get("ltp", 0) if pe else 0
                
                # Math Greeks evaluations via engine calculators
                ce_greeks = black_scholes_greeks(spot_price, strike_price, time_to_expiry_days, risk_free_rate, iv_override, "CE")
                pe_greeks = black_scholes_greeks(spot_price, strike_price, time_to_expiry_days, risk_free_rate, iv_override, "PE")
                
                chain_records.append({
                    "CE OI": ce_oi, "CE Delta": ce_greeks["delta"], "CE Theta": ce_greeks["theta"], "CE LTP": ce_ltp,
                    "Strike": strike_price,
                    "PE LTP": pe_ltp, "PE Theta": pe_greeks["theta"], "PE Delta": pe_greeks["delta"], "PE OI": pe_oi
                })
                
        df_chain = pd.DataFrame(chain_records).sort_values("Strike").reset_index(drop=True)
        
        # Summary Analytics Block
        total_ce_oi = df_chain["CE OI"].sum()
        total_pe_oi = df_chain["PE OI"].sum()
        pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
        
        # Trend and structural scoring setup
        sentiment = "Neutral Matrix"
        card_bg = "background: linear-gradient(135deg, #64748b, #475569);"
        if pcr >= 1.25:
            sentiment = "Strong Bullish Bias"
            card_bg = "background: linear-gradient(135deg, #15803d, #166534);"
        elif pcr > 1.05:
            sentiment = "Mildly Bullish Sentiment"
            card_bg = "background: linear-gradient(135deg, #22c55e, #15803d);"
        elif pcr <= 0.75:
            sentiment = "Strong Bearish Bias"
            card_bg = "background: linear-gradient(135deg, #b91c1c, #991b1b);"
        elif pcr < 0.95:
            sentiment = "Mildly Bearish Sentiment"
            card_bg = "background: linear-gradient(135deg, #ef4444, #b91c1c);"
            
        # ── Rendering Dashboard View Elements ──
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Underlying Spot Price", f"₹ {spot_price:,.2f}")
        m_col2.metric("Calculated ATM Level", f"{atm_strike}")
        m_col3.metric("Aggregate PCR (OI)", f"{pcr}")
        if adx_metrics:
            m_col4.metric("ADX (14 Period Trend)", f"{adx_metrics['adx']}", f"DI+/DI-: {adx_metrics['plus_di']}/{adx_metrics['minus_di']}")
        else:
            m_col4.metric("ADX (14 Period Trend)", "N/A")

        # HTML Sentiment Block Rendering
        st.markdown(f"""
        <div class="direction-card" style="{card_bg}">
            <div class="score-label">AUTOMATED STRUCTURAL TREND SIGNAL</div>
            <div class="direction-text">{sentiment}</div>
            <div class="sentiment-text">Total Put OI: {total_pe_oi:,.0f} contracts &nbsp;|&nbsp; Total Call OI: {total_ce_oi:,.0f} contracts</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ── Normal Distribution Price Prediction Engine ──
        st.subheader("🎯 Statistical Price Prediction Range (Normal Distribution Model)")
        
        std_dev_price = spot_price * iv_override * np.sqrt(time_to_expiry_days)
        lower_1sigma = spot_price - std_dev_price
        upper_1sigma = spot_price + std_dev_price
        lower_2sigma = spot_price - (2 * std_dev_price)
        upper_2sigma = spot_price + (2 * std_dev_price)
        
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("1-Sigma Low Bound (68.2%)", f"₹ {lower_1sigma:,.2f}")
        p_col2.metric("🎯 Median Expected Spot", f"₹ {spot_price:,.2f}")
        p_col3.metric("1-Sigma High Bound (68.2%)", f"₹ {upper_1sigma:,.2f}")
        
        fig_bell, ax_bell = plt.subplots(figsize=(12, 4.5))
        x_axis = np.linspace(spot_price - 3.5 * std_dev_price, spot_price + 3.5 * std_dev_price, 500)
        y_axis = norm.pdf(x_axis, spot_price, std_dev_price)
        
        ax_bell.plot(x_axis, y_axis, color="#0f172a", linewidth=2, label="Probability Density Function")
        ax_bell.fill_between(x_axis, y_axis, where=(x_axis >= lower_1sigma) & (x_axis <= upper_1sigma), 
                             color="#38bdf8", alpha=0.35, label="68.2% Confidence Zone (1σ)")
        ax_bell.fill_between(x_axis, y_axis, where=((x_axis >= lower_2sigma) & (x_axis < lower_1sigma)) | ((x_axis > upper_1sigma) & (x_axis <= upper_2sigma)), 
                             color="#0284c7", alpha=0.15, label="95.4% Confidence Zone (2σ)")
        
        ax_bell.axvline(spot_price, color="#0f172a", linestyle="-", linewidth=1.5, label=f"Current Spot ({spot_price:,.1f})")
        ax_bell.axvline(lower_1sigma, color="#ef4444", linestyle="--", linewidth=1.2)
        ax_bell.axvline(upper_1sigma, color="#22c55e", linestyle="--", linewidth=1.2)
        
        ax_bell.set_title(f"Expiry Statistical Forecast Structure for {selected_index_name} (Target: {selected_expiry})", fontsize=11, fontweight="bold")
        ax_bell.set_xlabel("Predicted Index Settlement Price", fontsize=9)
        ax_bell.set_ylabel("Probability Density", fontsize=9)
        ax_bell.legend(loc="upper right", fontsize=8)
        ax_bell.grid(True, linestyle=":", alpha=0.4)
        st.pyplot(fig_bell)
        
        # Display Core Options Execution Table Matrix Grid Array
        st.subheader("📊 Interactive Options Chain Model & Analytical Greeks")
        
        def color_strikes(row):
            val = row["Strike"]
            styles = [""] * len(row)
            if val < spot_price:
                styles = ["background-color: #f0fdf4;"] * len(row)
            if val > spot_price:
                styles = ["background-color: #fef2f2;"] * len(row)
            return styles
            
        styled_df = df_chain.style.apply(color_strikes, axis=1).format({
            "CE OI": "{:,.0f}", "CE Delta": "{:.2f}", "CE Theta": "{:.2f}", "CE LTP": "₹{:.2f}",
            "Strike": "{:,.0f}",
            "PE LTP": "₹{:.2f}", "PE Theta": "{:.2f}", "PE Delta": "{:.2f}", "PE OI": "{:,.0f}"
        })
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Matplotlib visualization arrays for dashboard charting layout execution
        st.subheader("📈 Open Interest Distribution Profile")
        fig, ax = plt.subplots(figsize=(12, 4))
        width = diff * 0.35
        ax.bar(df_chain["Strike"] - width/2, df_chain["CE OI"], width, label="Call Options OI", color="#ef4444", alpha=0.85)
        ax.bar(df_chain["Strike"] + width/2, df_chain["PE OI"], width, label="Put Options OI", color="#22c55e", alpha=0.85)
        ax.axvline(spot_price, color="#64748b", linestyle="--", linewidth=1.5, label=f"Spot Price ({spot_price})")
        ax.set_xlabel("Strike Prices", fontsize=10)
        ax.set_ylabel("Open Interest Contracts", fontsize=10)
        ax.set_title("Open Interest (OI) Concentration Array around Spot", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)

        # Handle active refresh tracking timers
        if auto_refresh:
            time.sleep(10)
            st.rerun()

    except Exception as error_msg:
        st.error(f"Runtime Pipeline Interruption: {error_msg}")
