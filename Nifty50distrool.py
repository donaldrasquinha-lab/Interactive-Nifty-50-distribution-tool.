import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from scipy.stats import norm

# --- DASHBOARD LAYOUT CONFIGURATION ---
st.set_page_config(page_title="Multi-Asset Probability Dashboard", layout="wide")

st.title("📊 Multi-Asset Probability & Price Prediction Tool")
st.markdown("Select a hardcoded Nifty stock or search any BSE asset to run automated mathematical volatility modeling.")

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

# Dropdown 1: Select between Index, Hardcoded NSE Stocks, or Live BSE Search
asset_type = st.sidebar.selectbox("Select Asset Type", ["Indices", "NSE Stocks (Nifty 50)", "BSE Stocks (Live Search)"])

# Initialize dynamic tracking variables
target_instrument_key = None
selected_asset_name = ""

# --- SEGMENT 1: INDICES (HARDCODED) ---
if asset_type == "Indices":
    asset_options = {
        "Nifty 50": "NSE_INDEX|Nifty 50",
        "Nifty Bank": "NSE_INDEX|Nifty Bank",
        "BSE Sensex": "BSE_INDEX|SENSEX"
    }
    selected_asset_name = st.sidebar.selectbox("Choose Specific Index", list(asset_options.keys()))
    target_instrument_key = asset_options[selected_asset_name]

# --- SEGMENT 2: NSE STOCKS (COMPLETE NIFTY 50 HARDCODED) ---
elif asset_type == "NSE Stocks (Nifty 50)":
    asset_options = {
        "Adani Ports (ADANIPORTS)": "NSE_EQ|INE742F01042",
        "Adani Enterprises (ADANIENT)": "NSE_EQ|INE423A01024",
        "Apollo Hospitals (APOLLOHOSP)": "NSE_EQ|INE437A01024",
        "Asian Paints (ASIANPAINT)": "NSE_EQ|INE021A01026",
        "Axis Bank (AXISBANK)": "NSE_EQ|INE238A01034",
        "Bajaj Auto (BAJAJ-AUTO)": "NSE_EQ|INE917I01010",
        "Bajaj Finance (BAJFINANCE)": "NSE_EQ|INE296A01024",
        "Bajaj Finserv (BAJAJFINSV)": "NSE_EQ|INE918I01018",
        "Bharat Petroleum (BPCL)": "NSE_EQ|INE029A01011",
        "Bharti Airtel (BHARTIARTL)": "NSE_EQ|INE397D01024",
        "Britannia Industries (BRITANNIA)": "NSE_EQ|INE216A01030",
        "Cipla (CIPLA)": "NSE_EQ|INE059A01026",
        "Coal India (COALINDIA)": "NSE_EQ|INE522F01014",
        "Divi's Laboratories (DIVISLAB)": "NSE_EQ|INE361B01024",
        "Dr. Reddy's Laboratories (DRREDDY)": "NSE_EQ|INE089A01023",
        "Eicher Motors (EICHERMOT)": "NSE_EQ|INE066A01021",
        "Grasim Industries (GRASIM)": "NSE_EQ|INE047A01021",
        "HCL Technologies (HCLTECH)": "NSE_EQ|INE860A01027",
        "HDFC Bank (HDFCBANK)": "NSE_EQ|INE040A01034",
        "HDFC Life (HDFCLIFE)": "NSE_EQ|INE795G01014",
        "Hero MotoCorp (HEROMOTOCO)": "NSE_EQ|INE158A01026",
        "Hindalco Industries (HINDALCO)": "NSE_EQ|INE038A01020",
        "Hindustan Unilever (HINDUNILVR)": "NSE_EQ|INE030A01027",
        "ICICI Bank (ICICIBANK)": "NSE_EQ|INE090A01021",
        "ITC Limited (ITC)": "NSE_EQ|INE154A01025",
        "IndusInd Bank (INDUSINDBK)": "NSE_EQ|INE095A01012",
        "Infosys (INFY)": "NSE_EQ|INE009A01021",
        "JSW Steel (JSWSTEEL)": "NSE_EQ|INE019A01038",
        "Kotak Mahindra Bank (KOTAKBANK)": "NSE_EQ|INE237A01028",
        "Larsen & Toubro (LT)": "NSE_EQ|INE018A01030",
        "LTIMindtree (LTIM)": "NSE_EQ|INE214B01027",
        "Mahindra & Mahindra (M&M)": "NSE_EQ|INE101A01026",
        "Maruti Suzuki (MARUTI)": "NSE_EQ|INE585B01010",
        "NTPC Limited (NTPC)": "NSE_EQ|INE733E01010",
        "Nestle India (NESTLEIND)": "NESTLEIND", # Note: Special Upstox direct string tracking key fallback mapping
        "ONGC (ONGC)": "NSE_EQ|INE213A01029",
        "Power Grid Corporation (POWERGRID)": "NSE_EQ|INE752E01010",
        "Reliance Industries (RELIANCE)": "NSE_EQ|INE002A01018",
        "SBI Life Insurance (SBILIFE)": "NSE_EQ|INE123W01016",
        "Shriram Finance (SHRIRAMFIN)": "NSE_EQ|INE721A01013",
        "State Bank of India (SBIN)": "NSE_EQ|INE062A01020",
        "Sun Pharmaceutical (SUNPHARMA)": "NSE_EQ|INE044A01036",
        "TCS (TCS)": "NSE_EQ|INE467B01029",
        "Tata Consumer Products (TATACONSUM)": "NSE_EQ|INE192A01025",
        "Tata Motors (TATAMOTORS)": "NSE_EQ|INE155A01022",
        "Tata Steel (TATASTEEL)": "NSE_EQ|INE081A01020",
        "Tech Mahindra (TECHM)": "NSE_EQ|INE669C01036",
        "Titan Company (TITAN)": "NSE_EQ|INE280A01028",
        "UltraTech Cement (ULTRACEMCO)": "NSE_EQ|INE481G01011",
        "Wipro (WIPRO)": "NSE_EQ|INE075A01022"
    }
    selected_asset_name = st.sidebar.selectbox("Choose Specific NSE Stock", list(asset_options.keys()))
    target_instrument_key = asset_options[selected_asset_name]

# --- SEGMENT 3: BSE STOCKS (LIVE KEYWORD SEARCH VIA API) ---
else:
    bse_search = st.sidebar.text_input("Type Stock Name / Ticker for BSE", value="Reliance")
    
    if upstox_token and bse_search:
        try:
            # Query the Upstox active database master lookup endpoint
            search_url = f"https://api.upstox.com/v2/instruments/search?keyword={bse_search}"
            search_headers = {'Accept': 'application/json', 'Authorization': f'Bearer {upstox_token}'}
            search_res = requests.get(search_url, headers=search_headers)
            
            if search_res.status_code == 200:
                search_data = search_res.json().get('data', [])
                # Filter down results belonging explicitly to the BSE Cash equity segment
                bse_results = [item for item in search_data if item.get('instrument_type') == 'EQUITY' and item.get('exchange') == 'BSE']
                
                if bse_results:
                    # Create dictionary mapping unique description names to their true dynamic key strings
                    bse_map = {f"{item['name']} ({item['trading_symbol']})": item['instrument_key'] for item in bse_results[:15]}
                    selected_asset_name = st.sidebar.selectbox("Matching BSE Results Found", list(bse_map.keys()))
                    target_instrument_key = bse_map[selected_asset_name]
                else:
                    st.sidebar.warning("⚠️ No matching BSE equity tokens found. Using fallback.")
            else:
                st.sidebar.error("Error connecting to Upstox Master Lookup.")
        except Exception as e:
            st.sidebar.error(f"Search API Error: {str(e)}")
            
    # Safe fallback variables if token isn't filled out or lookup isn't ready
    if not target_instrument_key:
        selected_asset_name = "Simulated BSE Asset"
        target_instrument_key = "BSE_EQ|500325"

# Horizon mapping configuration parameter
time_horizon = st.sidebar.selectbox("Select Prediction Horizon", ["1 Week", "1 Month", "3 Months", "1 Year"])
dte_mapping = {"1 Week": 7, "1 Month": 30, "3 Months": 90, "1 Year": 365}
days_to_target = dte_mapping[time_horizon]

# --- LIVE MARKET DATA ENGINE ---
# --- LIVE MARKET DATA ENGINE ---
def fetch_upstox_live_data(token, instrument_key):
    """
    Queries Upstox v2 Market Quote Endpoint for the chosen asset key
    and safely parses active prices and volatility metrics.
    """
    import urllib.parse
    
    # URL-encode the instrument key to turn '|' into '%7C' so the API receives it cleanly
    encoded_key = urllib.parse.quote(instrument_key)
    url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_key}"
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        json_data = response.json()
        
        # Upstox shifts the key separator from '|' to ':' inside the returned data dictionary
        response_key = instrument_key.replace("|", ":")
        
        if 'data' in json_data:
            # Case-insensitive safety check to handle varying API return states safely
            data_payload = {k.upper(): v for k, v in json_data['data'].items()}
            search_target = response_key.upper()
            
            if search_target in data_payload:
                instrument_data = data_payload[search_target]
                spot = float(instrument_data['last_price'])
                
                # Dynamic asset baseline volatility shifts
                base_iv = 0.24 if "EQ" in search_target else 0.155
                iv = float(instrument_data.get('oi_interest', base_iv))
                
                if iv <= 0 or iv > 1.5:  
                    iv = base_iv
                    
                return spot, iv, f"Upstox Live Feed ({response_key})"
            else:
                # Debug helper to show you exactly what keys came back if it fails
                available_keys = list(json_data['data'].keys())
                raise KeyError(f"Target '{response_key}' missing. API returned keys: {available_keys}")
        else:
            raise KeyError("Malformed API response structure: 'data' property missing.")
    else:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

            
            # Dynamic asset standard baseline shifts
            base_iv = 0.24 if "BSE_EQ" in instrument_key or "NSE_EQ" in instrument_key else 0.155
            iv = float(instrument_data.get('oi_interest', base_iv))
            if iv <= 0 or iv > 1.5:  
                iv = base_iv
                
            return spot, iv, f"Upstox Live Feed ({response_key})"
        else:
            raise KeyError(f"Key '{response_key}' missing in returned data payload.")
    else:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

# Execution branch choice based on Token Presence
if upstox_token and target_instrument_key:
    try:
        spot_price, implied_vol, data_source = fetch_upstox_live_data(upstox_token, target_instrument_key)
        st.sidebar.success(f"✅ Connected to API Engine")
    except Exception as e:
        st.sidebar.error(f"❌ Pull Failed. Using baseline simulation numbers.")
        st.sidebar.code(f"Error details: {str(e)}")
        spot_price, implied_vol, data_source = (2450.00, 0.24, "Simulated Equity Base") if "EQ" in target_instrument_key else (22350.00, 0.155, "Simulated Index Base")
else:
    spot_price, implied_vol, data_source = (2450.00, 0.24, "Simulated Equity Base") if "EQ" in target_instrument_key else (22350.00, 0.155, "Simulated Index Base")

# --- VOLATILITY MATHEMATICS CALCULATIONS ---
standard_deviation = spot_price * implied_vol * np.sqrt(days_to_target / 365)

upper_1sig = spot_price + standard_deviation
lower_1sig = spot_price - standard_deviation
upper_2sig = spot_price + (2 * standard_deviation)
lower_2sig = spot_price - (2 * standard_deviation)

# --- VISUAL CARDS RENDERING ---
st.subheader(f"🔮 Prediction Profile: {selected_asset_name or 'Asset'} ({time_horizon} Outlook)")
st.caption(f"Engine Feed: Data sourced via **{data_source}**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Current Spot Value", value=f"₹{spot_price:,.2f}")
with col2:
    st.metric(label="Expected Price Swings (±1σ)", value=f"₹{standard_deviation:,.2f}")
with col3:
    st.metric(label="Annual Volatility Map (IV)", value=f"{implied_vol*100:.1f}%")

# --- THE CHART INTERFACE ENGINE ---
x_axis_prices = np.linspace(spot_price - (3.5 * standard_deviation), spot_price + (3.5 * standard_deviation), 500)
y_axis_probability = norm.pdf(x_axis_prices, spot_price, standard_deviation)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_axis_prices, y=y_axis_probability, mode='lines', name='Normal Distribution',
    line=dict(color='#00d2ff', width=3),
    hovertemplate="<b>Price Target:</b> ₹%{x:,.2f}<br><b>Density Weight:</b> %{y:.6f}<extra></extra>"
))

x_68 = x_axis_prices[(x_axis_prices >= lower_1sig) & (x_axis_prices <= upper_1sig)]
y_68 = y_axis_probability[(x_axis_prices >= lower_1sig) & (x_axis_prices <= upper_1sig)]

fig.add_trace(go.Scatter(
    x=np.concatenate([x_68, x_68[::-1]]), y=np.concatenate([y_68, np.zeros_like(y_68)]),
    fill='toself', fillcolor='rgba(0, 210, 255, 0.15)', line=dict(color='rgba(255,255,255,0)'),
    name='68% Range', hoverinfo='skip'
))

fig.add_vline(x=spot_price, line_width=2, line_dash="dash", line_color="#ffffff", annotation_text="Spot")
fig.add_vline(x=lower_1sig, line_width=1.5, line_dash="dot", line_color="#00ff88", annotation_text="-1σ Floor")
fig.add_vline(x=upper_1sig, line_width=1.5, line_dash="dot", line_color="#00ff88", annotation_text="+1σ Ceiling")

fig.update_layout(
    xaxis_title=f"Target Price Levels (₹)",
    yaxis_title="Statistical Probability Density",
    template="plotly_dark", showlegend=False, height=480
)

# Render with the modern width standard mapping configuration
st.plotly_chart(fig, width="stretch")

# --- PLAIN ENGLISH TRANSLATION SUMMARY PANEL ---
st.markdown("### 📋 Investor Insight Summary")
st.info(f"""
*   📊 **Probability Target Ranges:** Based on current calculations, there is a **68.2% mathematical probability** that **{selected_asset_name}** will trade within the range of **₹{lower_1sig:,.2f}** and **₹{upper_1sig:,.2f}** over the next {time_horizon}.
""")
