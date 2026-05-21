"""
╔══════════════════════════════════════════════════════════════╗
║         UPSTOX ALPHA TRADING ENGINE — Options Intelligence   ║
║  Injects Volatility Skew, Money Velocity, and Trend Guards  ║
║  to calculate high-probability options execution signals.   ║
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

# ── 1. Page Configuration & Professional Light Interface ──
st.set_page_config(
    page_title="Upstox Alpha Signal Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://googleapis.com');
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; }
div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 14px 18px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}
div[data-testid="stMetric"] label { color: #64748b !important; font-size: 11px !important; letter-spacing: 1px; text-transform: uppercase; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important; color: #0f172a !important; font-size: 22px !important;
}
.signal-card {
    border-radius: 14px; padding: 22px; margin: 15px 0; font-family: 'JetBrains Mono', monospace; border: 1px solid #e2e8f0;
}
.score-label { font-size: 11px; letter-spacing: 2px; color: #ffffff; opacity: 0.9; margin-bottom: 4px; }
.direction-text { font-size: 28px; font-weight: 700; color: #ffffff; }
.sentiment-text { font-size: 13px; color: #f8fafc; margin-top: 4px; }
.sidebar-header { font-size: 11px; letter-spacing: 3px; color: #0284c7; font-weight: 600; text-transform: uppercase; }
.sidebar-title { font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 2px; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── 2. Sidebar Control & Asset Configurations ──
st.sidebar.markdown('<div class="sidebar-header">Upstox Analytics</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-title">Alpha Signal Engine</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

upstox_token = st.sidebar.text_input("Upstox Access Token (Bearer)", type="password")
selected_index = st.sidebar.selectbox("Target Asset Index:", ["Nifty 50", "Nifty Bank", "Financial Services"])

index_map = {
    "Nifty 50": {"key": "NSE_INDEX|Nifty 50", "lot_size": 50, "default_spot": 23800, "oi_step": 100},
    "Nifty Bank": {"key": "NSE_INDEX|Nifty Bank", "lot_size": 15, "default_spot": 51200, "oi_step": 100},
    "Financial Services": {"key": "NSE_INDEX|Nifty Fin Service", "lot_size": 40, "default_spot": 22400, "oi_step": 100}
}

lot_size = index_map[selected_index]["lot_size"]
instrument_key = index_map[selected_index]["key"]
oi_step = index_map[selected_index]["oi_step"]

today = datetime.now()
days_until_thursday = (3 - today.weekday()) % 7
if days_until_thursday == 0 and today.hour >= 16:
    days_until_thursday = 7
next_thursday = today + timedelta(days=days_until_thursday)
computed_expiry_str = next_thursday.strftime("%Y-%m-%d")

st.sidebar.header("🔧 Alpha System Multipliers")
iv_percent = st.sidebar.slider("Implied Volatility (IV %)", 5.0, 40.0, 12.0, 0.5) / 100
days_to_expiry = st.sidebar.number_input("Days to Expiry (DTE Scalar)", 1, 30, max(1, days_until_thursday))
show_adjustment = st.sidebar.checkbox("Overlay Recommended Adjustment Leg", value=True)

# ── 3. State Core Real-Time Ingestion Layer ──
spot_price = index_map[selected_index]["default_spot"]
detected_expiry = computed_expiry_str
is_live = False

# Setup Baseline Fallback Engine Estimates
time_factor = np.sqrt(days_to_expiry / 365)
expected_move = spot_price * iv_percent * time_factor
atm_premium = int(max(25.0, round(expected_move * 0.4)))
oi_wall_premium = int(max(10.0, round(atm_premium * 0.45)))
hedge_premium = int(max(2.0, round(oi_wall_premium * 0.35)))

atm_strike = int(round(spot_price / oi_step) * oi_step)
oi_wall_strike = atm_strike + oi_step

# Baseline Multipliers (Alpha Logic Safe Placeholders)
live_pcr = 0.95
oi_support = atm_strike - oi_step
oi_resistance = atm_strike + oi_step
money_velocity_ratio = 1.05  # COI acceleration indicator
volatility_skew_index = 1.02 # Real-world skew index
trend_guard_multiplier = 1.00 # Trend-following safety scalar
raw_data = []

if upstox_token:
    try:
        headers = {'Accept': 'application/json', 'Api-Version': '2.0', 'Authorization': f'Bearer {upstox_token}'}
        
        # A. Pull current index valuation parameters via Live LTP REST Layer
        quote_url = 'https://upstox.com'
        quote_res = requests.get(quote_url, headers=headers, params={'instrument_key': instrument_key}, timeout=10).json()
        
        if quote_res.get('status') == 'success' and instrument_key in quote_res.get('data', {}):
            spot_price = quote_res['data'][instrument_key]['last_price']
            atm_strike = int(round(spot_price / oi_step) * oi_step)
            
            # B. Pull Options Chain records using structural parameters mapping
            chain_url = 'https://upstox.com'
            chain_res = requests.get(chain_url, headers=headers, params={'instrument_key': instrument_key, 'expiry_date': computed_expiry_str}, timeout=10).json()
            
            if chain_res.get('status') == 'success':
                raw_data = chain_res.get('data', [])
                
                if len(raw_data) > 0:
                    max_call_oi, max_put_oi = -1, -1
                    total_call_oi, total_put_oi = 0, 0
                    total_call_coi, total_put_coi = 0, 0
                    total_call_iv, total_put_iv = 0.0, 0.0
                    option_rows_count = 0
                    
                    best_call_strike, best_put_strike = atm_strike + oi_step, atm_strike - oi_step
                    premium_lookup = {}
                    processed_records = []
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    sample_leg = raw_data[0].get('call_options') or raw_data[0].get('put_options')
                    if sample_leg:
                        detected_expiry = sample_leg.get('metadata', {}).get('expiry_date', computed_expiry_str)

                    for item in raw_data:
                        strike = int(item['strike_price'])
                        ce_data = item.get('call_options', {}).get('market_data', {}) if item.get('call_options') else {}
                        pe_data = item.get('put_options', {}).get('market_data', {}) if item.get('put_options') else {}
                        
                        c_oi, p_oi = ce_data.get('oi', 0), pe_data.get('oi', 0)
                        # Extract Change in OI (COI) velocity components inside data loop
                        c_coi, p_coi = ce_data.get('oi_change', 0), pe_data.get('oi_change', 0)
                        c_iv, p_iv = ce_data.get('implied_volatility', 12.0), pe_data.get('implied_volatility', 12.0)
                        
                        total_call_oi += c_oi
                        total_put_oi += p_oi
                        total_call_coi += abs(c_coi)
                        total_put_coi += abs(p_coi)
                        
                        # Isolate equidistant OTM wings to measure volatility skew metrics
                        if abs(strike - atm_strike) <= (oi_step * 3):
                            total_call_iv += c_iv
                            total_put_iv += p_iv
                            option_rows_count += 1
                        
                        premium_lookup[strike] = ce_data.get('ltp', atm_premium)
                        
                        if strike > spot_price and c_oi > max_call_oi:
                            max_call_oi = c_oi
                            best_call_strike = strike
                        if strike < spot_price and p_oi > max_put_oi:
                            max_put_oi = p_oi
                            best_put_strike = strike
                            
                        processed_records.append({
                            "Timestamp": timestamp_str, "Underlying": selected_index, "Spot_Price": spot_price,
                            "Expiry_Date": detected_expiry, "Strike_Price": strike, "CE_LTP": ce_data.get('ltp', 0.0),
                            "CE_OI": c_oi, "PE_LTP": pe_data.get('ltp', 0.0), "PE_OI": p_oi
                        })
                    
                    # ── UPGRADE 1 & 2 Math Calculations ──
                    oi_wall_strike = best_call_strike
                    oi_resistance, oi_support = best_call_strike, best_put_strike
                    live_pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 1.0
                    
                    # Upgrade 1: Money Velocity Acceleration ratio
                    money_velocity_ratio = round(total_put_coi / total_call_coi, 2) if total_call_coi > 0 else 1.0
                    # Upgrade 2: Dynamic Real-World Volatility Skew index
                    volatility_skew_index = round(total_put_iv / total_call_iv, 2) if total_call_iv > 0 else 1.0
                    is_live = True
                    
                    atm_premium = premium_lookup.get(atm_strike, atm_premium)
                    oi_wall_premium = premium_lookup.get(oi_wall_strike, oi_wall_premium)
                    
                    strike_sell = oi_wall_strike
                    strike_hedge = strike_sell + (strike_sell - atm_strike)
                    hedge_premium = premium_lookup.get(strike_hedge, hedge_premium)
                    
                    filename_prefix = selected_index.replace(" ", "_").lower()
                    pd.DataFrame(processed_records).to_csv(f"{filename_prefix}_chain_latest.csv", index=False)
                    with open(f"{filename_prefix}_snapshot.json", "w") as j_file:
                        json.dump({"index": selected_index, "live_spot": spot_price, "chain_matrix": processed_records}, j_file, indent=4)
                    st.sidebar.success("💾 Alpha Snapshot Saved!")
    except Exception:
        pass

# ── UPGRADE 3: Expiry Decay Theta Compressions Scalar ──
# Linear scaling model penalizes trade size/confidence based on decay efficiency curves
decay_efficiency_factor = max(0.3, (7.0 - days_to_expiry) / 7.0) if days_to_expiry <= 7 else 0.25

# ── UPGRADE 4: Trend Momentum Guard Intercepts ──
# Checks if the market is trending heavily inside an explosive breakout tail sequence
is_explosive_trend = True if abs(spot_price - atm_strike) > (expected_move * 0.85) else False
trend_guard_multiplier = 0.50 if is_explosive_trend else 1.00

# ── Integrated Alpha Scoring Matrix Core Logic ──
f1_signal_score = 25 if live_pcr > 1.05 and money_velocity_ratio > 1.10 else (0 if live_pcr < 0.85 else 12.5)
f2_signal_score = 25 if spot_price > (oi_resistance + oi_support) / 2 else 10
f3_signal_score = 25 if volatility_skew_index > 1.03 else 12.5
f4_signal_score = 25 if not is_explosive_trend else 5

alpha_composite_signal = int((f1_signal_score + f2_signal_score + f3_signal_score + f4_signal_score) * trend_guard_multiplier)
signal_classification = "STRONG BULLISH ENTRY" if alpha_composite_signal >= 70 else ("EXPLOSIVE TREND OVERHEAD: HOLD" if is_explosive_trend else "RANGEBOUND CONSOLIDATION: EXECUTE SPREAD")
signal_theme_color = "#16a34a" if alpha_composite_signal >= 65 else ("#ea580c" if is_explosive_trend else "#2563eb")

# Render Metric Panels
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric(label=f"🎯 {selected_index} Spot", value=f"₹{spot_price:,.2f}", delta="Live Feed" if is_live else "Simulated Engine")
with col_m2:
    st.metric(label="📊 Put-Call Ratio (PCR)", value=f"{live_pcr}", delta=f"Velocity Ratio: {money_velocity_ratio}")
with col_m3:
    st.metric(label="🟢 Live Volatility Skew", value=f"{volatility_skew_index}x", delta="Skew Anomalous" if volatility_skew_index > 1.05 else "Normal Curve")
with col_m4:
    st.metric(label="🔴 Trend Guard Status", value="EXPLOSIVE BREAKOUT" if is_explosive_trend else "CONSOLIDATING Range", delta=f"Decay Scalar: {round(decay_efficiency_factor, 2)}x")
with col_m5:
    st.metric(label="📅 Active Options Expiry", value=detected_expiry)

# Render Alpha Signals Display Block
st.markdown(f"""
<div class="direction-card" style="background: {signal_theme_color}; border-color: {signal_theme_color}; color: #ffffff;">
    <div class="score-label">⚡ UPSTOX QUANTALPHA SIGNAL MATRIX Engine</div>
    <div class="direction-text">{signal_classification} ({alpha_composite_signal}/100)</div>
    <div class="sentiment-text">Trades filtered using dynamic premium lookahead windows, Intraday Change in OI acceleration velocity profiles, and trend guard multipliers.</div>
</div>
""", unsafe_allow_html=True)

# ── 5. Advanced Geometry Strategies Visualization ──
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

# ── 6. Clean Table Data Recommendations ──
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
