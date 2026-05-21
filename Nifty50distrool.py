import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import requests
import json
import pandas as pd
from datetime import datetime

# 1. Page Configuration & Theme Setup
st.set_page_config(page_title="Upstox Multi-Index Data Logger", layout="wide")
st.title("⚡ Upstox Live Analytical Data Logger & Dashboard")
st.markdown("---")

# 2. Sidebar Control Panel & Token Configuration
st.sidebar.header("🔑 Authentication & Setup")
upstox_token = st.sidebar.text_input(
    "Upstox Access Token (Bearer)", 
    type="password", 
    help="Paste your morning authenticated API access token here."
)

st.sidebar.header("🎯 Asset Configuration")
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

# Volatility Controls for Statistical Calculations
st.sidebar.header("🔧 Statistical Parameters")
iv_percent = st.sidebar.slider("Implied Volatility (IV %)", min_value=5.0, max_value=40.0, value=12.0, step=0.5) / 100
days_to_expiry = st.sidebar.number_input("Days to Expiry (For SD Calculation)", min_value=1, max_value=30, value=7)

st.sidebar.header("🛠️ Risk Controls")
show_adjustment = st.sidebar.checkbox("Overlay Recommended Adjustment Leg", value=True)

# 3. State Management & Live Data Handlers
spot_price = index_map[selected_index]["default_spot"]
detected_expiry = "Simulation_Expiry"
is_live = False

# Setup baseline mathematical defaults for option premiums
time_factor = np.sqrt(days_to_expiry / 365)
expected_move = spot_price * iv_percent * time_factor
atm_premium = int(max(25.0, round(expected_move * 0.4)))
oi_wall_premium = int(max(10.0, round(atm_premium * 0.45)))
hedge_premium = int(max(2.0, round(oi_wall_premium * 0.35)))

atm_strike = int(round(spot_price / oi_step) * oi_step)
oi_wall_strike = atm_strike + oi_step

if upstox_token:
    try:
        headers = {
            'Accept': 'application/json',
            'Api-Version': '2.0',
            'Authorization': f'Bearer {upstox_token}'
        }
        
        # A. Fetch Spot Price using the dedicated LTP Quotes Endpoint
        quote_url = 'https://upstox.com'
        quote_params = {'instrument_key': instrument_key}
        quote_response = requests.get(quote_url, headers=headers, params=quote_params)
        
        if quote_response.status_code == 200:
            quote_res = quote_response.json()
            if quote_res.get('status') == 'success' and instrument_key in quote_res.get('data', {}):
                spot_price = quote_res['data'][instrument_key]['last_price']
                atm_strike = int(round(spot_price / oi_step) * oi_step)
                is_live = True
                
        # B. Fetch Option Chain and Extract Expiry Date & Premiums
        chain_url = 'https://upstox.com'
        chain_params = {'instrument_key': instrument_key} 
        chain_response = requests.get(chain_url, headers=headers, params=chain_params)
        
        if chain_response.status_code == 200:
            chain_res = chain_response.json()
            raw_data = chain_res.get('data', [])
            
            if chain_res.get('status') == 'success' and len(raw_data) > 0:
                max_oi = -1
                best_oi_strike = atm_strike + oi_step
                premium_lookup = {}
                processed_records = []
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Detect target expiry date dynamically from the first valid row payload
                if len(raw_data) > 0:
                    first_row = raw_data[0]
                    sample_leg = first_row.get('call_options') or first_row.get('put_options')
                    if sample_leg:
                        detected_expiry = sample_leg.get('metadata', {}).get('expiry_date', 'Unknown')

                # Step C: Loop through chain to populate lookups and computational log matrices
                for item in raw_data:
                    strike = int(item['strike_price'])
                    ce_data = item.get('call_options', {}).get('market_data', {}) if item.get('call_options') else {}
                    pe_data = item.get('put_options', {}).get('market_data', {}) if item.get('put_options') else {}
                    
                    current_oi = ce_data.get('oi', 0)
                    premium_lookup[strike] = ce_data.get('ltp', atm_premium)
                    
                    # Track and isolate the highest Call Open Interest (OI) strike wall
                    if strike > spot_price and current_oi > max_oi:
                        max_oi = current_oi
                        best_oi_strike = strike
                        
                    # Structure data packets for files logging
                    processed_records.append({
                        "Timestamp": timestamp_str,
                        "Underlying": selected_index,
                        "Spot_Price": spot_price,
                        "Expiry_Date": detected_expiry,
                        "Strike_Price": strike,
                        "CE_LTP": ce_data.get('ltp', 0.0),
                        "CE_OI": current_oi,
                        "PE_LTP": pe_data.get('ltp', 0.0),
                        "PE_OI": pe_data.get('oi', 0)
                    })
                
                # Assign dynamic metrics parsed from live stream
                oi_wall_strike = best_oi_strike
                atm_premium = premium_lookup.get(atm_strike, atm_premium)
                oi_wall_premium = premium_lookup.get(oi_wall_strike, oi_wall_premium)
                
                strike_sell = oi_wall_strike
                strike_hedge = strike_sell + (strike_sell - atm_strike)
                hedge_premium = premium_lookup.get(strike_hedge, hedge_premium)
                
                # Write Computation Data Files Directly to Disk Server Space
                filename_prefix = selected_index.replace(" ", "_").lower()
                
                # 1. Output computational CSV matrix file
                df_export = pd.DataFrame(processed_records)
                df_export.to_csv(f"{filename_prefix}_chain_latest.csv", index=False)
                
                # 2. Output computational nested JSON file package
                json_package = {
                    "index": selected_index, "live_spot": spot_price, "active_expiry": detected_expiry,
                    "last_synced": timestamp_str, "chain_matrix": processed_records
                }
                with open(f"{filename_prefix}_snapshot.json", "w") as j_file:
                    json.dump(json_package, j_file, indent=4)
                    
                st.sidebar.success("💾 Core Snapshot Files Logged!")
        else:
            st.sidebar.warning(f"Option Chain API Offline ({chain_response.status_code}). Using emulation model.")
    except Exception as e:
        st.sidebar.error(f"API Extractor Error: {str(e)}")

# Define target geometry coordinates for final strategy plotting matrices
strike_buy = atm_strike
strike_sell = oi_wall_strike
strike_hedge = strike_sell + (strike_sell - strike_buy)

# 4. Mathematical Engine Setup (68-95-99.7 Rule)
one_sd_move = spot_price * iv_percent * time_factor
sd1_lower, sd1_upper = spot_price - one_sd_move, spot_price + one_sd_move
sd2_lower, sd2_upper = spot_price - (2 * one_sd_move), spot_price + (2 * one_sd_move)

# Order Quantities
qty_buy = 1
qty_sell = 2
qty_hedge = 1

# Generate Expiry Price Range Grid Arrays
x = np.linspace(spot_price - (3 * one_sd_move), spot_price + (3 * one_sd_move), 2000)

# Calculate Core Strategy Payoffs
payoff_buy = (np.maximum(x - strike_buy, 0) - atm_premium) * qty_buy
payoff_sell = (oi_wall_premium - np.maximum(x - strike_sell, 0)) * qty_sell
y_initial = (payoff_buy + payoff_sell) * lot_size

payoff_hedge = (np.maximum(x - strike_hedge, 0) - hedge_premium) * qty_hedge
y_adjusted = y_initial + (payoff_hedge * lot_size)

# Strategy Break-even Boundaries
lower_be = strike_buy + (atm_premium - (2 * oi_wall_premium))
upper_be = strike_sell + ((strike_sell - strike_buy) - (atm_premium - (2 * oi_wall_premium)))

# Display Connection Banner Status Ribbon
if is_live:
    st.success(f"🟢 LIVE API FEED MODE • Index: {selected_index} | Isolated Expiry: {detected_expiry}")
else:
    st.warning(f"🟡 SIMULATION MODE (No Token Given) • Displaying Approximated Parametric Curves for {selected_index}")

# 5. Dual Dashboard Plot Layout
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"### 📊 Probability Distribution Curve")
    prob_density = norm.pdf(x, spot_price, one_sd_move)
    fig_p, ax_p = plt.subplots(figsize=(10, 4.5))
    ax_p.plot(x, prob_density, color='#475569', linewidth=2)
    
    # Fill standard deviation confidence bands
    ax_p.fill_between(x, prob_density, 0, where=(x >= sd1_lower) & (x <= sd1_upper), facecolor='#10b981', alpha=0.3, label='68.2% (1 SD)')
    ax_p.fill_between(x, prob_density, 0, where=((x >= sd2_lower) & (x < sd1_lower)) | ((x > sd1_upper) & (x <= sd2_upper)), facecolor='#f59e0b', alpha=0.2, label='95.4% (2 SD)')
    
    ax_p.axvline(spot_price, color='#6366f1', linestyle=':', linewidth=1.5, label=f'Spot: ₹{int(spot_price)}')
    ax_p.axvline(strike_sell, color='#dc2626', linestyle='--', linewidth=1.2, label=f'OI Wall Strike: ₹{strike_sell}')
    ax_p.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax_p.set_xlim(x.min(), x.max())
    ax_p.get_yaxis().set_visible(False)
    st.pyplot(fig_p)

with col_right:
    st.markdown("### 📈 Strategy Expiry Payoff Matrix (Comparative View)")
    fig_t, ax_t = plt.subplots(figsize=(10, 4.5))
    ax_t.plot(x, y_initial, color="#14b8a6", linewidth=2.5, linestyle="--", alpha=0.6, label="Initial Ratio Call Spread")
    
    if show_adjustment:
        ax_t.plot(x, y_adjusted, color="#2563eb", linewidth=3, label="Adjusted Configuration (Risk Capped)")
        ax_t.fill_between(x, y_adjusted, 0, where=(y_adjusted >= 0), facecolor='#10b981', alpha=0.15)
        ax_t.fill_between(x, y_adjusted, 0, where=(y_adjusted < 0), facecolor='#ef4444', alpha=0.15)
        
        max_prof_y = np.max(y_adjusted)
        max_prof_x = x[np.argmax(y_adjusted)]
        ax_t.scatter(max_prof_x, max_prof_y, color='#1d4ed8', s=100, zorder=5)
        ax_t.text(max_prof_x - 140, max_prof_y - 950, f"Max Profit: ₹{int(max_prof_y)}", fontsize=9, weight='bold', color='#1d4ed8')
    else:
        ax_t.fill_between(x, y_initial, 0, where=(y_initial >= 0), facecolor='#10b981', alpha=0.2)
        ax_t.fill_between(x, y_initial, 0, where=(y_initial < 0), facecolor='#ef4444', alpha=0.2)
        
        max_prof_y = np.max(y_initial)
        max_prof_x = x[np.argmax(y_initial)]
        ax_t.scatter(max_prof_x, max_prof_y, color='#0f766e', s=100, zorder=5)
        ax_t.text(max_prof_x - 140, max_prof_y - 950, f"Max Profit: ₹{int(max_prof_y)}", fontsize=9, weight='bold', color='#0f766e')

    ax_t.axhline(0, color='#475569', linestyle='-', linewidth=1.2)
    # SYNTAX CORRECTED HERE: Replaced raw space comma with an explicit coordinate matching baseline array [0, 0]
    ax_t.scatter([lower_be, upper_be], [0, 0], color='#b45309', s=60, zorder=5)
    ax_t.set_xlim(x.min(), x.max())
    ax_t.grid(True, linestyle=":", alpha=0.5)
    st.pyplot(fig_t)

# 6. Execution Matrix and Dynamic Order Output Data Tables
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
