"""
╔══════════════════════════════════════════════════════════════╗
║         UPSTOX OI ANALYZER & STATISTICAL DASHBOARD          ║
║  Pulls live OI data, analyzes CE/PE buildup around ATM,     ║
║  graphs distribution curves, and plots spread adjustments.  ║
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

# ── 1. Page Configuration & Professional Styling ──
st.set_page_config(
    page_title="Upstox Pro OI Analyzer & Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS variables mimicking professional trading terminals
st.markdown("""
<style>
@import url('https://googleapis.com');

/* Global Style Foundations */
.stApp { background: #0a0e17; }
section[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1e293b; }
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; color: #f8fafc !important; }

/* Dynamic KPI metric panels */
div[data-testid="stMetric"] {
    background: #111827; border: 1px solid #1e293b; border-radius: 12px;
    padding: 14px 18px;
}
div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 11px !important; letter-spacing: 1px; text-transform: uppercase; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important;
    color: #f1f5f9 !important; font-size: 22px !important;
}

/* Direction sentiment block */
.direction-card {
    border-radius: 14px; padding: 20px; margin: 15px 0;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid #1e293b;
}
.score-label { font-size: 11px; letter-spacing: 2px; color: #94a3b8; margin-bottom: 4px; }
.direction-text { font-size: 26px; font-weight: 700; color: #f8fafc; }
.sentiment-text { font-size: 13px; color: #cbd5e1; margin-top: 4px; }

/* Header titles */
.sidebar-header { font-size: 11px; letter-spacing: 3px; color: #38bdf8; font-weight: 600; text-transform: uppercase; }
.sidebar-title { font-size: 20px; font-weight: 700; color: #f1f5f9; margin-top: 2px; }

/* Hide Streamlit default interface anchors */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── 2. Sidebar Control Panel & Token Configuration ──
st.sidebar.markdown('<div class="sidebar-header">Upstox Analytics</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-title">OI Engine Pro</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

upstox_token = st.sidebar.text_input(
    "Upstox Access Token (Bearer)", 
    type="password", 
    help="Paste your morning authenticated API access token here."
)

selected_index = st.sidebar.selectbox(
    "Select Target Index:",
    ["Nifty 50", "Nifty Bank", "Financial Services"]
)

# Index Mapping Configurations (Keys matching Upstox Instrument Master)
index_map = {
    "Nifty 50": {"key": "NSE_INDEX|Nifty 50", "lot_size": 50, "default_spot": 23800, "oi_step": 100},
    "Nifty Bank": {"key": "NSE_INDEX|Nifty Bank", "lot_size": 15, "default_spot": 51200, "oi_step": 100},
    "Financial Services": {"key": "NSE_INDEX|Nifty Fin Service", "lot_size": 40, "default_spot": 22400, "oi_step": 100}
}

lot_size = index_map[selected_index]["lot_size"]
instrument_key = index_map[selected_index]["key"]
oi_step = index_map[selected_index]["oi_step"]

# DYNAMIC EXPIRY CALCULATION: Finds the next upcoming Thursday naturally
today = datetime.now()
days_until_thursday = (3 - today.weekday()) % 7
if days_until_thursday == 0 and today.hour >= 16: # If today is Thursday after market hours, roll to next week
    days_until_thursday = 7
next_thursday = today + timedelta(days=days_until_thursday)
computed_expiry_str = next_thursday.strftime("%Y-%m-%d")

# Volatility Controls for Statistical Calculations
st.sidebar.header("🔧 Statistical Parameters")
iv_percent = st.sidebar.slider("Implied Volatility (IV %)", min_value=5.0, max_value=40.0, value=12.0, step=0.5) / 100
days_to_expiry = st.sidebar.number_input("Days to Expiry (For SD Calculation)", min_value=1, max_value=30, value=max(1, days_until_thursday))

st.sidebar.header("🛠️ Risk Controls")
show_adjustment = st.sidebar.checkbox("Overlay Recommended Adjustment Leg", value=True)

# ── 3. State Management & Live Data Handlers ──
spot_price = index_map[selected_index]["default_spot"]
detected_expiry = computed_expiry_str
is_live = False

# Setup baseline simulation/fallback defaults
time_factor = np.sqrt(days_to_expiry / 365)
expected_move = spot_price * iv_percent * time_factor
atm_premium = int(max(25.0, round(expected_move * 0.4)))
oi_wall_premium = int(max(10.0, round(atm_premium * 0.45)))
hedge_premium = int(max(2.0, round(oi_wall_premium * 0.35)))

atm_strike = int(round(spot_price / oi_step) * oi_step)
oi_wall_strike = atm_strike + oi_step

# Baseline Simulation KPI Variables
live_pcr = 0.95
oi_support = atm_strike - oi_step
oi_resistance = atm_strike + oi_step
raw_data = []

if upstox_token:
    try:
        headers = {
            'Accept': 'application/json',
            'Api-Version': '2.0',
            'Authorization': f'Bearer {upstox_token}'
        }
        
        # A. Fetch Spot Price using the dedicated LTP Quotes Endpoint [1]
        quote_url = 'https://upstox.com'
        quote_params = {'instrument_key': instrument_key}
        quote_response = requests.get(quote_url, headers=headers, params=quote_params, timeout=10)
        
        if quote_response.status_code == 200 and 'application/json' in quote_response.headers.get('Content-Type', ''):
            quote_res = quote_response.json()
            
            # API Safety Check Guard [1]
            if quote_res.get('status') == 'success' and instrument_key in quote_res.get('data', {}):
                spot_price = quote_res['data'][instrument_key]['last_price']
                atm_strike = int(round(spot_price / oi_step) * oi_step)
                
                # B. Fetch Option Chain ONLY if Spot Price step succeeded cleanly [1]
                chain_url = 'https://upstox.com'
                chain_params = {'instrument_key': instrument_key, 'expiry_date': computed_expiry_str} 
                chain_response = requests.get(chain_url, headers=headers, params=chain_params, timeout=10)
                
                if chain_response.status_code == 200 and 'application/json' in chain_response.headers.get('Content-Type', ''):
                    chain_res = chain_response.json()
                    raw_data = chain_res.get('data', [])
                    
                    if chain_res.get('status') == 'success' and len(raw_data) > 0:
                        max_call_oi = -1
                        max_put_oi = -1
                        total_call_oi = 0
                        total_put_oi = 0
                        
                        best_call_strike = atm_strike + oi_step
                        best_put_strike = atm_strike - oi_step
                        
                        premium_lookup = {}
                        processed_records = []
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Detect target expiry date dynamically from payload metadata [1]
                        first_row = raw_data[0]
                        sample_leg = first_row.get('call_options') or first_row.get('put_options')
                        if sample_leg:
                            detected_expiry = sample_leg.get('metadata', {}).get('expiry_date', computed_expiry_str)

                        # Loop through option chain array matrices [1]
                        for item in raw_data:
                            strike = int(item['strike_price'])
                            ce_data = item.get('call_options', {}).get('market_data', {}) if item.get('call_options') else {}
                            pe_data = item.get('put_options', {}).get('market_data', {}) if item.get('put_options') else {}
                            
                            current_call_oi = ce_data.get('oi', 0)
                            current_put_oi = pe_data.get('oi', 0)
                            
                            total_call_oi += current_call_oi
                            total_put_oi += current_put_oi
                            
                            premium_lookup[strike] = ce_data.get('ltp', atm_premium)
                            
                            # Track Support and Resistance via Highest Put/Call OI walls [1]
                            if strike > spot_price and current_call_oi > max_call_oi:
                                max_call_oi = current_call_oi
                                best_call_strike = strike
                                
                            if strike < spot_price and current_put_oi > max_put_oi:
                                max_put_oi = current_put_oi
                                best_put_strike = strike
                                
                            processed_records.append({
                                "Timestamp": timestamp_str,
                                "Underlying": selected_index,
                                "Spot_Price": spot_price,
                                "Expiry_Date": detected_expiry,
                                "Strike_Price": strike,
                                "CE_LTP": ce_data.get('ltp', 0.0),
                                "CE_OI": current_call_oi,
                                "PE_LTP": pe_data.get('ltp', 0.0),
                                "PE_OI": current_put_oi
                            })
                        
                        # Dynamically assign Live Calculated Metrics [1]
                        oi_wall_strike = best_call_strike
                        oi_resistance = best_call_strike
                        oi_support = best_put_strike
                        live_pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 1.0
                        is_live = True
                        
                        atm_premium = premium_lookup.get(atm_strike, atm_premium)
                        oi_wall_premium = premium_lookup.get(oi_wall_strike, oi_wall_premium)
                        
                        strike_sell = oi_wall_strike
                        strike_hedge = strike_sell + (strike_sell - atm_strike)
                        hedge_premium = premium_lookup.get(strike_hedge, hedge_premium)
                        
                        # Save Data Files to Disk Space [1]
                        filename_prefix = selected_index.replace(" ", "_").lower()
                        df_export = pd.DataFrame(processed_records)
                        df_export.to_csv(f"{filename_prefix}_chain_latest.csv", index=False)
                        
                        json_package = {
                            "index": selected_index, "live_spot": spot_price, "active_expiry": detected_expiry,
                            "last_synced": timestamp_str, "chain_matrix": processed_records
                        }
                        with open(f"{filename_prefix}_snapshot.json", "w") as j_file:
                            json.dump(json_package, j_file, indent=4)
            else:
                st.sidebar.error("Upstox platform token rejected. Loading simulation model engine.")
        else:
            st.sidebar.error(f"Quote Connection Denied ({quote_response.status_code}). check Token string.")
    except Exception as e:
        st.sidebar.error(f"API Feed offline. Reverting to structural fallback modeling parameters.")

# Generate emulated data row metrics if live stream connectivity is inactive
if not is_live:
    atm_strike = int(round(spot_price / oi_step) * oi_step)
    oi_resistance = atm_strike + oi_step
    oi_support = atm_strike - oi_step
    oi_wall_strike = oi_resistance
    
    # Mathematical emulation constants [1]
    total_pe_oi_sim = 24500000
    total_ce_oi_sim = 22000000
    live_pcr = round(total_pe_oi_sim / total_ce_oi_sim, 2)

# ── 4. 4-Factor Sentiment Scoring Model ──
# Factor 1: Put-Call Ratio distribution bias [1]
f1_score = 25 if live_pcr > 1.05 else (0 if live_pcr < 0.85 else 12.5)
# Factor 2: Position of spot relative to macro boundaries [1]
f2_score = 25 if spot_price > (oi_resistance + oi_support) / 2 else 10
# Factor 3: Open Interest dominance profile [1]
f3_score = 25 if live_pcr >= 1.0 else 5
# Factor 4: Probability envelope proximity (Inside 1 SD) [1]
f4_score = 25 if abs(spot_price - atm_strike) < (spot_price * iv_percent * time_factor) else 5

final_sentiment_score = int(f1_score + f2_score + f3_score + f4_score)
direction_label = "BULLISH" if final_sentiment_score >= 65 else ("BEARISH" if final_sentiment_score <= 40 else "NEUTRAL")
card_bg_color = "#064e3b" if direction_label == "BULLISH" else ("#7f1d1d" if direction_label == "BEARISH" else "#1e293b")

# Render KPI Metric Panel Blocks [1]
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric(label=f"🎯 {selected_index} Spot", value=f"₹{spot_price:,.2f}", delta="Live" if is_live else "Simulated")
with col_m2:
    st.metric(label="📊 Put-Call Ratio (PCR)", value=f"{live_pcr}", delta="Bullish (>1)" if live_pcr >= 1.0 else "Bearish (<1)", delta_color="normal" if live_pcr >= 1.0 else "inverse")
with col_m3:
    st.metric(label="🟢 OI Support Floor", value=f"₹{oi_support:,}")
with col_m4:
    st.metric(label="🔴 OI Resistance Wall", value=f"₹{oi_resistance:,}", delta_color="inverse")
with col_m5:
    st.metric(label="📅 Target Expiry Date", value=detected_expiry)

# Render Custom Direction Sentiment Block [1]
st.markdown(f"""
<div class="direction-card" style="background: {card_bg_color};">
    <div class="score-label">4-FACTOR DIRECTIONAL MARKETS BIAS</div>
    <div class="direction-text">{direction_label} ({final_sentiment_score}/100)</div>
    <div class="sentiment-text">Calculated using live Put-Call buildup profiles [1], key open interest walls [1], and standard normal price probability variations.</div>
</div>
""", unsafe_allow_html=True)

# ── 5. Math Strategy & Graphical Payload Layout ──
strike_buy = atm_strike
strike_sell = oi_wall_strike
strike_hedge = strike_sell + (strike_sell - strike_buy)

one_sd_move = spot_price * iv_percent * time_factor
sd1_lower, sd1_upper = spot_price - one_sd_move, spot_price + one_sd_move
sd2_lower, sd2_upper = spot_price - (2 * one_sd_move), spot_price + (2 * one_sd_move)

qty_buy, qty_sell, qty_hedge = 1, 2, 1

# Expiry Price range matrix coordinates
x = np.linspace(spot_price - (3 * one_sd_move), spot_price + (3 * one_sd_move), 2000)

payoff_buy = (np.maximum(x - strike_buy, 0) - atm_premium) * qty_buy
payoff_sell = (oi_wall_premium - np.maximum(x - strike_sell, 0)) * qty_sell
y_initial = (payoff_buy + payoff_sell) * lot_size

payoff_hedge = (np.maximum(x - strike_hedge, 0) - hedge_premium) * qty_hedge
y_adjusted = y_initial + (payoff_hedge * lot_size)

lower_be = strike_buy + (atm_premium - (2 * oi_wall_premium))
upper_be = strike_sell + ((strike_sell - strike_buy) - (atm_premium - (2 * oi_wall_premium)))

# Render Dual Graphs Side-by-Side
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"### 📊 Normal Distribution (68-95-99.7 Rule Curve)")
    prob_density = norm.pdf(x, spot_price, one_sd_move)
    fig_p, ax_p = plt.subplots(figsize=(10, 4.5))
    fig_p.patch.set_facecolor('#0a0e17')
    ax_p.set_facecolor('#111827')
    
    ax_p.plot(x, prob_density, color='#64748b', linewidth=2)
    ax_p.fill_between(x, prob_density, 0, where=(x >= sd1_lower) & (x <= sd1_upper), facecolor='#10b981', alpha=0.25, label='68.2% Confidence Zone (1 SD)')
    ax_p.fill_between(x, prob_density, 0, where=((x >= sd2_lower) & (x < sd1_lower)) | ((x > sd1_upper) & (x <= sd2_upper)), facecolor='#f59e0b', alpha=0.15, label='95.4% Confidence Zone (2 SD)')
    
    ax_p.axvline(spot_price, color='#6366f1', linestyle=':', linewidth=1.5, label=f'Spot Price Level')
    ax_p.axvline(strike_sell, color='#f43f5e', linestyle='--', linewidth=1.2, label=f'OI Strike Barrier')
    
    ax_p.tick_params(colors='#94a3b8', labelsize=9)
    ax_p.legend(loc="upper left", frameon=True, facecolor='#111827', edgecolor='#1e293b', labelcolor='#e2e8f0', fontsize=8)
    ax_p.set_xlim(x.min(), x.max())
    ax_p.get_yaxis().set_visible(False)
    st.pyplot(fig_p)

with col_right:
    st.markdown("### 📈 Derivatives Strategy Expiry Payoff Matrix")
    fig_t, ax_t = plt.subplots(figsize=(10, 4.5))
    fig_t.patch.set_facecolor('#0a0e17')
    ax_t.set_facecolor('#111827')
    
    ax_t.plot(x, y_initial, color="#14b8a6", linewidth=2, linestyle="--", alpha=0.6, label="Initial Ratio Call Spread")
    
    if show_adjustment:
        ax_t.plot(x, y_adjusted, color="#2563eb", linewidth=2.5, label="Adjusted Configuration (Risk Capped)")
        ax_t.fill_between(x, y_adjusted, 0, where=(y_adjusted >= 0), facecolor='#10b981', alpha=0.12)
        ax_t.fill_between(x, y_adjusted, 0, where=(y_adjusted < 0), facecolor='#ef4444', alpha=0.12)
        
        max_prof_y = np.max(y_adjusted)
        max_prof_x = x[np.argmax(y_adjusted)]
        ax_t.scatter(max_prof_x, max_prof_y, color='#3b82f6', s=80, zorder=5)
    else:
        ax_t.fill_between(x, y_initial, 0, where=(y_initial >= 0), facecolor='#10b981', alpha=0.15)
        ax_t.fill_between(x, y_initial, 0, where=(y_initial < 0), facecolor='#ef4444', alpha=0.15)
        
        max_prof_y = np.max(y_initial)
        max_prof_x = x[np.argmax(y_initial)]
        ax_t.scatter(max_prof_x, max_prof_y, color='#14b8a6', s=80, zorder=5)

    ax_t.axhline(0, color='#475569', linestyle='-', linewidth=1.2)
    ax_t.scatter([lower_be, upper_be],, color='#f59e0b', s=50, zorder=5)
    
    ax_t.tick_params(colors='#94a3b8', labelsize=9)
    ax_t.legend(loc="upper left", frameon=True, facecolor='#111827', edgecolor='#1e293b', labelcolor='#e2e8f0', fontsize=8)
    ax_t.set_xlim(x.min(), x.max())
    ax_t.grid(True, linestyle=":", alpha=0.1)
    st.pyplot(fig_t)

# ── 6. Order Output Data Tables ──
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
