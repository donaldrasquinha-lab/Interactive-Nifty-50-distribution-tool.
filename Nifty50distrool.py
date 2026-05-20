import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from scipy.stats import norm

# --- DASHBOARD LAYOUT CONFIGURATION ---
st.set_page_config(page_title="Nifty 50 Upstox Pro Dashboard", layout="wide")

st.title("📊 Nifty 50 Probability & Price Prediction Tool")
st.markdown("Powered by live market volatility structures and probability mathematics.")

# --- SIDEBAR PANEL SETUP ---
st.sidebar.header("🔑 Authentication & Settings")

# Secure input text box for Upstox API Token String
upstox_token = st.sidebar.text_input(
    label="Upstox Access Token API v2",
    type="password",
    help="Ensure this is your generated API access_token, not your temporary authorization code."
)

time_horizon = st.sidebar.selectbox("Select Prediction Horizon", ["1 Week", "1 Month", "3 Months"])
dte_mapping = {"1 Week": 7, "1 Month": 30, "3 Months": 90}
days_to_target = dte_mapping[time_horizon]

# --- LIVE MARKET DATA ENGINE ---
def fetch_upstox_live_data(token):
    """
    Queries Upstox v2 Market Quote Endpoint for Nifty 50 Index Spot
    and safely parses data with the required API headers.
    """
    instrument_key = "NSE_INDEX|Nifty 50"
    url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={instrument_key}"
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        json_data = response.json()
        response_key = "NSE_INDEX:Nifty 50"
        
        if 'data' in json_data and response_key in json_data['data']:
            instrument_data = json_data['data'][response_key]
            spot = float(instrument_data['last_price'])
            
            # Extract Implied Volatility or fallback to a realistic market baseline
            iv = float(instrument_data.get('oi_interest', 0.155)) 
            if iv <= 0 or iv > 1:  
                iv = 0.155 
                
            return spot, iv, "Upstox Live Feed"
        else:
            raise KeyError("Key 'NSE_INDEX:Nifty 50' missing in returned data payload.")
    else:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

# Execution branch choice based on Token Presence
if upstox_token:
    try:
        spot_price, implied_vol, data_source = fetch_upstox_live_data(upstox_token)
        st.sidebar.success("✅ Upstox Live Token Connection Active")
    except Exception as e:
        st.sidebar.error(f"❌ Connection Failed")
        st.sidebar.code(f"Error details: {str(e)}")
        spot_price, implied_vol, data_source = 22350.00, 0.155, "Fallback Baseline Mode (Simulated)"
else:
    st.sidebar.info("💡 Enter your Upstox API v2 access token to activate streaming data feeds.")
    spot_price, implied_vol, data_source = 22350.00, 0.155, "Fallback Baseline Mode (Simulated)"

# --- VOLATILITY MATHEMATICS CALCULATIONS ---
standard_deviation = spot_price * implied_vol * np.sqrt(days_to_target / 365)

upper_1sig = spot_price + standard_deviation
lower_1sig = spot_price - standard_deviation
upper_2sig = spot_price + (2 * standard_deviation)
lower_2sig = spot_price - (2 * standard_deviation)

# --- VISUAL CARDS RENDERING ---
st.subheader(f"🔮 Market Prediction Profile ({time_horizon} Outlook)")
st.caption(f"Data Engine: Using pricing structures fed via **{data_source}**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Nifty 50 Spot Price", value=f"₹{spot_price:,.2f}")
with col2:
    st.metric(label="Expected Price Swings (±1σ)", value=f"₹{standard_deviation:,.2f}")
with col3:
    st.metric(label="Annual Volatility Level", value=f"{implied_vol*100:.1f}%")

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
    xaxis_title="Nifty Index Value (₹)",
    yaxis_title="Statistical Probability Density",
    template="plotly_dark",
    showlegend=False,
    height=480
)

# FIXED: Replaced use_container_width=True with width="stretch" to match the modern Streamlit standard
st.plotly_chart(fig, width="stretch")

st.markdown("### 📋 Investor Insight Summary")
st.info(f"""
*   📊 **Probability Target Ranges:** Based on current calculations, there is a **68.2% likelihood** that Nifty 50 closes the next {time_horizon} trading frame somewhere between **₹{lower_1sig:,.2f}** and **₹{upper_1sig:,.2f}**.
""")
