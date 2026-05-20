import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- DASHBOARD CONFIGURATION ---
st.set_page_config(page_title="Nifty 50 Smart Probability Dashboard", layout="wide")

st.title("📊 Nifty 50 Probability & Price Prediction Tool")
st.markdown("Designed for everyday investors to understand market expectations and risk at a glance.")

# --- STEP 1: GET LIVE DATA & ESTIMATE VOLATILITY ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour to keep it fast
def fetch_live_market_data():
    # Fetch Nifty Spot Price from Yahoo Finance
    nifty = yf.Ticker("^NSEI")
    spot = nifty.history(period="1d")['Close'].iloc[-1]
    
    # Grab options expirations and pick the near-month (Monthly Expiry proxy)
    expiries = nifty.options
    if len(expiries) > 1:
        chosen_expiry = expiries[1]  # Pick the next near-month contract
    else:
        chosen_expiry = expiries[0]
        
    chain = nifty.option_chain(chosen_expiry)
    
    # Calculate a proxy for Implied Volatility using At-the-Money options
    atm_calls = chain.calls[(chain.calls['strike'] >= spot - 100) & (chain.calls['strike'] <= spot + 100)]
    avg_iv = atm_calls['impliedVolatility'].mean() if not atm_calls.empty else 0.15
    
    return spot, chosen_expiry, avg_iv

try:
    spot_price, expiry_date, implied_vol = fetch_live_market_data()
except Exception as e:
    st.error("Could not fetch real-time market data. Using fallback baseline figures.")
    spot_price, expiry_date, implied_vol = 22000.0, "Next Monthly Expiry", 0.15

# --- STEP 2: USER CONTROLS IN SIDEBAR ---
st.sidebar.header("🔧 Settings")
time_horizon = st.sidebar.selectbox("Select Prediction Horizon", ["1 Week", "1 Month", "3 Months"])

# Map human selection to calendar days
dte_mapping = {"1 Week": 7, "1 Month": 30, "3 Months": 90}
days_to_target = dte_mapping[time_horizon]

# --- STEP 3: CALCULATE PROBABILITIES ---
# Standard deviation formula: Price * IV * sqrt(t/365)
standard_deviation = spot_price * implied_vol * np.sqrt(days_to_target / 365)

upper_1sig = spot_price + standard_deviation
lower_1sig = spot_price - standard_deviation
upper_2sig = spot_price + (2 * standard_deviation)
lower_2sig = spot_price - (2 * standard_deviation)

# --- STEP 4: VISUAL LAYPERSON KPI CARDS ---
st.subheader(f"🔮 Market Prediction for the Next {time_horizon} (By {expiry_date})")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Current Nifty 50 Price", value=f"₹{spot_price:,.2f}")
with col2:
    st.metric(label="Expected Price Swings (±)", value=f"₹{standard_deviation:,.2f}", 
              help="Calculated using options market Implied Volatility. The market expects Nifty to swing by this amount.")
with col3:
    st.metric(label="Market Volatility (IV)", value=f"{implied_vol*100:.1f}%", 
              help="Higher percentage means the market is anticipating larger, riskier price movements.")

# --- STEP 5: THE GRAPH (PLOTLY BELL CURVE) ---
# Generate X axis data points (price range to display on graph)
x_axis_prices = np.linspace(spot_price - (3.5 * standard_deviation), spot_price + (3.5 * standard_deviation), 500)
# Generate normal distribution Y axis curve
y_axis_probability = norm.pdf(x_axis_prices, spot_price, standard_deviation)

fig = go.Figure()

# Plot the main Normal Distribution Curve line
fig.add_trace(go.Scatter(
    x=x_axis_prices, y=y_axis_probability,
    mode='lines', name='Probability Curve',
    line=dict(color='#1f77b4', width=3),
    hovertemplate="<b>Price Target:</b> ₹%{x:,.2f}<br><b>Probability Density:</b> %{y:.6f}<extra></extra>"
))

# Shade the 68% High-Probability Safe Zone
x_68 = x_axis_prices[(x_axis_prices >= lower_1sig) & (x_axis_prices <= upper_1sig)]
y_68 = y_axis_probability[(x_axis_prices >= lower_1sig) & (x_axis_prices <= upper_1sig)]

fig.add_trace(go.Scatter(
    x=np.concatenate([x_68, x_68[::-1]]),
    y=np.concatenate([y_68, np.zeros_like(y_68)]),
    fill='toself', fillcolor='rgba(46, 204, 113, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='68% Safe Expected Zone',
    hoverinfo='skip'
))

# Add vertical dashed lines for clarity
fig.add_vline(x=spot_price, line_width=2, line_dash="dash", line_color="white", annotation_text="Current Price")
fig.add_vline(x=lower_1sig, line_width=1.5, line_dash="dot", line_color="#2ecc71", annotation_text="-1σ Floor")
fig.add_vline(x=upper_1sig, line_width=1.5, line_dash="dot", line_color="#2ecc71", annotation_text="+1σ Ceiling")

fig.update_layout(
    title=f"Where will Nifty 50 land in {time_horizon}? (Statistical Bell Curve)",
    xaxis_title="Nifty 50 Price Levels (₹)",
    yaxis_title="Relative Probability",
    template="plotly_dark",
    showlegend=False,
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# --- STEP 6: PLAIN ENGLISH TRANSLATION ---
st.markdown("### 📋 Plain English Breakdown for Investors")
st.info(f"""
*   🟢 **The 68% Safe Zone:** There is a **68.2% mathematical chance** that Nifty 50 will stay between **₹{lower_1sig:,.2f}** and **₹{upper_1sig:,.2f}** over the next {time_horizon}.
*   🟡 **The Extreme Zones (Outliers):** There is only an approximate **16% chance** that Nifty rises spectacularly above **₹{upper_1sig:,.2f}**, and a **16% chance** it drops completely below **₹{lower_1sig:,.2f}**. 
*   🔴 **The Black Swan Boundary:** A move beyond **₹{lower_2sig:,.2f}** or **₹{upper_2sig:,.2f}** is highly unlikely (less than a 5% total chance). If you are selling options, your safety strikes should be chosen beyond these red boundaries.
""")
