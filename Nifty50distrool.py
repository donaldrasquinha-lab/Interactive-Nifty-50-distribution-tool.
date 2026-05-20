import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from scipy.stats import norm

# --- DASHBOARD LAYOUT CONFIGURATION ---
st.set_page_config(page_title="Multi-Asset Probability Dashboard", layout="wide")

st.title("📊 Multi-Asset Probability & Price Prediction Tool")
st.markdown("Select an index or individual equity stock to run automated mathematical volatility modeling.")

# --- SIDEBAR PANEL SETUP ---
st.sidebar.header("🔑 Authentication & Settings")

# Secure input text box for Upstox API Token String
upstox_token = st.sidebar.text_input(
    label="Upstox Access Token API v2",
    type="password",
    help="Ensure this is your generated API access_token, not your temporary authorization code."
)

st.sidebar.markdown("---")
st.sidebar.header("📈 Asset Selection")

# Dropdown 1: Select between Index or Individual Shares
asset_type = st.sidebar.selectbox("Select Asset Type", ["Indices", "NSE Stocks", "BSE Stocks"])

# Map human-readable dropdown options to Upstox API system instrument keys
if asset_type == "Indices":
    asset_options = {
        "Nifty 50": "NSE_INDEX|Nifty 50",
        "Nifty Bank": "NSE_INDEX|Nifty Bank",
        "BSE Sensex": "BSE_INDEX|SENSEX"
    }
elif asset_type == "NSE Stocks":
    asset_options = {
        "Reliance Industries (NSE)": "NSE_EQ|RELIANCE",
        "TCS (NSE)": "NSE_EQ|TCS",
        "HDFC Bank (NSE)": "NSE_EQ|HDFCBANK",
        "Infosys (NSE)": "NSE_EQ|INFY",
        "Tata Motors (NSE)": "NSE_EQ|TATAMOTORS"
    }
else:  # BSE Stocks (Using Upstox standard numeric token strings)
    asset_options = {
        "Reliance Industries (BSE)": "BSE_EQ|500325",
        "TCS (BSE)": "BSE_EQ|532540",
        "HDFC Bank (BSE)": "BSE_EQ|500180",
        "State Bank of India (BSE)": "BSE_EQ|500112"
    }

# Dropdown 2: Dynamic selector based on the chosen asset type
selected_asset_name = st.sidebar.selectbox("Choose Specific Asset", list(asset_options.keys()))
target_instrument_key = asset_options[selected_asset_name]

# Dropdown 3: Time Frame Configuration
time_horizon = st.sidebar.selectbox("Select Prediction Horizon", ["1 Week", "1 Month", "3 Months", "1 Year"])
dte_mapping = {"1 Week": 7, "1 Month": 30, "3 Months": 90, "1 Year": 365}
days_to_target = dte_mapping[time_horizon]

# --- LIVE MARKET DATA ENGINE ---
def fetch_upstox_live_data(token, instrument_key):
    """
    Queries Upstox v2 Market Quote Endpoint for the chosen asset key
    and safely parses active prices and volatility metrics.
    """
    # Encode special pipe character if necessary for raw URLs
    url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={instrument_key}"
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        json_data = response.json()
        
        # Upstox switches the dictionary key string format from '|' to ':' inside the JSON body return
        response_key = instrument_key.replace("|", ":")
        
        if 'data' in json_data and response_key in json_data['data']:
            instrument_data = json_data['data'][response_key]
            spot = float(instrument_data['last_price'])
            
            # Default Volatility assignment logic based on asset classes
            # Stocks generally manifest a higher volatility base profile than indices
            base_iv = 0.24 if "_EQ" in instrument_key else 0.155
            
            # Try to grab open interest volatility proxy or fallback to baseline numbers
            iv = float(instrument_data.get('oi_interest', base_iv)) 
            if iv <= 0 or iv > 1.5:  
                iv = base_iv
                
            return spot, iv, f"Upstox Live Feed ({response_key})"
        else:
            raise KeyError(f"Key '{response_key}' missing in returned data payload.")
    else:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

# Execution branch choice based on Token Presence
if upstox_token:
    try:
        spot_price, implied_vol, data_source = fetch_upstox_live_data(upstox_token, target_instrument_key)
        st.sidebar.success(f"✅ Active Connection: {selected_asset_name}")
    except Exception as e:
        st.sidebar.error(f"❌ Connection Failed. Using simulated data.")
        st.sidebar.code(f"Error details: {str(e)}")
        # Default mock baseline logic if credentials mismatch
        spot_price, implied_vol, data_source = (2450.00, 0.24, "Simulated Equity Baseline") if "_EQ" in target_instrument_key else (22350.00, 0.155, "Simulated Index Baseline")
else:
    st.sidebar.info("💡 Enter your Upstox API v2 token to unlock active data streaming pipelines.")
    spot_price, implied_vol, data_source = (2450.00, 0.24, "Simulated Equity Baseline") if "_EQ" in target_instrument_key else (22350.00, 0.155, "Simulated Index Baseline")

# --- VOLATILITY MATHEMATICS CALCULATIONS ---
standard_deviation = spot_price * implied_vol * np.sqrt(days_to_target / 365)

upper_1sig = spot_price + standard_deviation
lower_1sig = spot_price - standard_deviation
upper_2sig = spot_price + (2 * standard_deviation)
lower_2sig = spot_price - (2 * standard_deviation)

# --- VISUAL CARDS RENDERING ---
st.subheader(f"🔮 Market Prediction Profile: {selected_asset_name} ({time_horizon} Outlook)")
st.caption(f"Engine Feed: Data sourced via **{data_source}**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Current Spot Value", value=f"₹{spot_price:,.2f}")
with col2:
    st.metric(label="Expected Price Swings (±1σ)", value=f"₹{standard_deviation:,.2f}",
              help="The range boundary expected to hold 68.2% of outcomes over this duration.")
with col3:
    st.metric(label="Annual Volatility Map (IV)", value=f"{implied_vol*100:.1f}%",
              help="Higher percentages yield flatter curves, indicating wider anticipated swing corridors.")

# --- THE CHART INTERFACE ENGINE ---
x_axis_prices = np.linspace(spot_price - (3.5 * standard_deviation), spot_price + (3.5 * standard_deviation), 500)
y_axis_probability = norm.pdf(x_axis_prices, spot_price, standard_deviation)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_axis_prices, y=y_axis_probability,
    mode='lines', name='Normal Distribution',
    line=dict(color='#00d2ff', width=3),
    hovertemplate="<b>Price Target:</b> ₹%{x:,.2f}<br><b>Density Weight:</b> %{y:.6f}<extra></extra>"
))

x_68 = x_axis_prices[(x_axis_prices >= lower_1sig) & (x_axis_prices <= upper_1sig)]
y_68 = y_axis_probability[(x_axis_prices >= lower_1sig) & (x_axis_prices <= upper_1sig)]

fig.add_trace(go.Scatter(
    x=np.concatenate([x_68, x_68[::-1]]),
    y=np.concatenate([y_68, np.zeros_like(y_68)]),
    fill='toself', fillcolor='rgba(0, 210, 255, 0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    name='68% Range',
    hoverinfo='skip'
))

fig.add_vline(x=spot_price, line_width=2, line_dash="dash", line_color="#ffffff", annotation_text="Spot")
fig.add_vline(x=lower_1sig, line_width=1.5, line_dash="dot", line_color="#00ff88", annotation_text="-1σ Floor")
fig.add_vline(x=upper_1sig, line_width=1.5, line_dash="dot", line_color="#00ff88", annotation_text="+1σ Ceiling")

fig.update_layout(
    xaxis_title=f"{selected_asset_name} Target Price (₹)",
    yaxis_title="Statistical Probability Density",
    template="plotly_dark",
    showlegend=False,
    height=480
)

# Future-proof modern container scaling syntax
st.plotly_chart(fig, width="stretch")

# --- PLAIN ENGLISH TRANSLATION SUMMARY PANEL ---
st.markdown("### 📋 Investor Insight Summary")
st.info(f"""
*   📊 **Probability Target Ranges:** Based on current volatility configurations, there is a **68.2% mathematical probability** that **{selected_asset_name}** will trade within the range of **₹{lower_1sig:,.2f}** and **₹{upper_1sig:,.2f}** over the next {time_horizon}.
*   🛡️ **Risk Boundary Mapping:** Moves sliding lower than **₹{lower_2sig:,.2f}** or spiking past **₹{upper_2sig:,.2f}** represent outliers with less than a 5% historical probability density profile. Options traders look to write premium strategies well outside these lines.
""")
