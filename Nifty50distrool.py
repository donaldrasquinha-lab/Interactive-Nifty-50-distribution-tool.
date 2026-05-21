import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. Page Configuration & UI Theme Setup
st.set_page_config(page_title="Institutional OI & Adjustment Dashboard", layout="wide")
st.title("⚡ Live Institutional OI Dashboard with Dynamic Adjustments")
st.markdown("---")

# 2. Live Market Feed Environment (Nifty 50)
spot_price = 23800
lot_size = 50

# Sidebar Metrics Tracking Panel
st.sidebar.header("🔧 Live Market Parameters")
st.sidebar.metric(label="Nifty 50 Live Spot", value=f"₹{spot_price}", delta=" +0.45% (Live)")
st.sidebar.metric(label="Highest Call OI Wall Strike", value="24000 CE", delta="Strong Resistance Cluster", delta_color="inverse")

st.sidebar.header("🛠️ Risk Mitigation Overlay")
show_adjustment = st.sidebar.checkbox("Overlay Recommended Adjustment Leg", value=True)

# 3. Strategy Parameters & Math Architecture
strike_buy = 23800
premium_buy = 160
qty_buy = 1

strike_sell = 24000
premium_sell = 70
qty_sell = 2

# Adjustment Leg Configuration (Scenario: Bought 24200 CE to cap naked risk)
strike_hedge = 24200
premium_hedge = 25
qty_hedge = 1

# Generate Expiry Underlying Price Ranges for Plotting
x = np.linspace(23400, 24500, 1500)

# Calculate Core Mathematical Payoffs
payoff_buy = (np.maximum(x - strike_buy, 0) - premium_buy) * qty_buy
payoff_sell = (premium_sell - np.maximum(x - strike_sell, 0)) * qty_sell
y_initial = (payoff_buy + payoff_sell) * lot_size

payoff_hedge = (np.maximum(x - strike_hedge, 0) - premium_hedge) * qty_hedge
y_adjusted = y_initial + (payoff_hedge * lot_size)

# Calculate Key Strategic Break-even Boundaries
lower_be = strike_buy + (premium_buy - (qty_sell * premium_sell))  # 23820
upper_be = strike_sell + ((strike_sell - strike_buy) - (premium_buy - (qty_sell * premium_sell)))  # 24180

# 4. Display Live Recommendations Tables Side-by-Side
st.subheader("📋 Executable Order Matrix & Live Recommendations")
col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.markdown("### 🟢 Phase 1: Recommended Initial Setup")
    initial_data = {
        "Action": ["🟢 BUY (ATM Strike)", "🔴 SELL (OI Wall Strike)"],
        "Option Strike": [f"{strike_buy} CE", f"{strike_sell} CE"],
        "Lots / Qty": [f"1 Lot ({lot_size} Qty)", f"2 Lots ({lot_size * 2} Qty)"],
        "Live Premium": [f"₹{premium_buy}", f"₹{premium_sell}"],
        "Net Premium Flow": [f"-₹{premium_buy * lot_size}", f"+₹{premium_sell * qty_sell * lot_size}"]
    }
    st.table(initial_data)
    st.info(f"**Max Capital Risk (Premium Outflow):** ₹{abs((qty_sell * premium_sell) - premium_buy) * lot_size} if market drops completely.")

with col_rec2:
    st.markdown("### 🟠 Phase 2: Recommended Adjustment Leg")
    if not show_adjustment:
        st.warning("⚠️ Market is within safe statistical boundaries. Adjustment layer is currently **INACTIVE**.")
        st.markdown("**Automated Trigger Rule:** Execute if Index breaches **₹24,000** and heavy Short Covering is detected.")
    else:
        st.success("🔥 **ADJUSTMENT ACTIVATED** (Spot breached the OI Wall)")
        adj_data = {
            "Action": ["🟢 BUY (OTM Hedge Option)"],
            "Option Strike": [f"{strike_hedge} CE"],
            "Lots / Qty": [f"1 Lot ({lot_size} Qty)"],
            "Live Premium": [f"₹{premium_hedge}"],
            "Net Margin Impact": [f"-₹{premium_hedge * lot_size}"]
        }
        st.table(adj_data)
        st.info("**Strategy State:** Structure successfully modified from unlimited risk to a safe **Iron Butterfly** framework.")

# 5. Visual Visualization Engine (Dual Plotting Architecture)
st.subheader("📊 Live Strategy Payoff Engine (Comparative View)")

fig, ax = plt.subplots(figsize=(11, 4.8))

# Plot initial profile
ax.plot(x, y_initial, color="#14b8a6", linewidth=2.5, linestyle="--", alpha=0.7, label="Initial 1:2 Ratio Call Spread")

if show_adjustment:
    # Plot adjusted profile
    ax.plot(x, y_adjusted, color="#2563eb", linewidth=3, label="Adjusted Structure (Risk Capped)")
    ax.fill_between(x, y_adjusted, 0, where=(y_adjusted >= 0), facecolor='#10b981', alpha=0.15, label='Adjusted Profit Zone')
    ax.fill_between(x, y_adjusted, 0, where=(y_adjusted < 0), facecolor='#ef4444', alpha=0.15, label='Adjusted Loss Zone')
    
    # Calculate and plot adjusted peak profit
    max_prof_y = np.max(y_adjusted)
    max_prof_x = x[np.argmax(y_adjusted)]
    ax.scatter(max_prof_x, max_prof_y, color='#1d4ed8', s=100, zorder=5)
    ax.text(max_prof_x - 140, max_prof_y - 950, f"Adjusted Max Profit: ₹{int(max_prof_y)}", fontsize=9, weight='bold', color='#1d4ed8')
else:
    # Fill background on initial strategy if toggle is off
    ax.fill_between(x, y_initial, 0, where=(y_initial >= 0), facecolor='#10b981', alpha=0.2, label='Initial Profit Zone')
    ax.fill_between(x, y_initial, 0, where=(y_initial < 0), facecolor='#ef4444', alpha=0.2, label='Initial Loss Zone')
    
    # Calculate and plot initial peak profit
    max_prof_y = np.max(y_initial)
    max_prof_x = x[np.argmax(y_initial)]
    ax.scatter(max_prof_x, max_prof_y, color='#0f766e', s=100, zorder=5)
    ax.text(max_prof_x - 140, max_prof_y - 950, f"Initial Max Profit: ₹{int(max_prof_y)}", fontsize=9, weight='bold', color='#0f766e')

# Strategic Reference Anchors
ax.axhline(0, color='#475569', linestyle='-', linewidth=1.2)
ax.axvline(spot_price, color='#6366f1', linestyle=':', linewidth=1.5, label=f'Live Spot (₹{spot_price})')
ax.axvline(strike_sell, color='#dc2626', linestyle=':', linewidth=1.5, label=f'OI Wall Resistance (₹{strike_sell})')

# Annotating Break-Even Points for Safety Mapping
ax.scatter([lower_be, upper_be], [0, 0], color='#b45309', s=60, zorder=5)
ax.text(lower_be - 150, 400, f'BE1: {lower_be}', fontsize=8.5, color='#b45309', weight='bold')
ax.text(upper_be + 20, 400, f'BE2: {upper_be}', fontsize=8.5, color='#b45309', weight='bold')

# Formatting labels & Canvas Settings
ax.set_title("Nifty 50 Comparative Expiry Payoff Matrix", fontsize=11, weight='bold', pad=10)
ax.set_xlabel("Nifty 50 Underlying Index Level", fontsize=9)
ax.set_ylabel("Net Payoff Return (₹)", fontsize=9)
ax.legend(loc="upper left", frameon=True, fontsize=8.5)
ax.set_xlim(23400, 24500)
ax.grid(True, linestyle=":", alpha=0.5)

st.pyplot(fig)
