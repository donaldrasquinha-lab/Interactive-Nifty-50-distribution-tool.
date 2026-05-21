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
import urllib.parse
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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
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

/* Strategy execution playbook blocks */
.playbook-card {
    background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px; padding: 18px; margin: 10px 0;
}
.playbook-title { font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }

/* Max pain & support/resistance badges */
.sr-badge {
    display: inline-block; padding: 4px 12px; border-radius: 6px; font-size: 12px;
    font-weight: 600; margin: 2px 4px; font-family: 'JetBrains Mono', monospace;
}
.sr-support { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.sr-resist  { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.sr-maxpain { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }

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

# ── Upstox V2 API Base ──
UPSTOX_BASE = "https://api.upstox.com/v2"


# ── Upstox API Helper ──
class UpstoxClient:
    """Wrapper around Upstox V2 REST API endpoints."""

    def __init__(self, token: str):
        clean_token = token.strip().replace("Bearer ", "")
        self.headers = {
            "Authorization": f"Bearer {clean_token}",
            "Accept": "application/json",
        }

    # ── Response safety ──
    def _safe_json(self, response):
        """Safely parses response bodies, avoiding string crashes on raw HTML pages."""
        ct = response.headers.get("Content-Type", "").lower()
        if "application/json" not in ct:
            raise ValueError(
                f"The API returned a non-JSON response (likely an HTML login page). "
                f"This usually means the Access Token has expired (tokens expire at midnight IST daily). "
                f"Status {response.status_code}. Snippet: {response.text[:200]}"
            )
        body = response.json()
        if body.get("status") == "error":
            errors = body.get("errors", [])
            msg = errors[0].get("message", str(errors)) if errors else str(body)
            raise ValueError(f"Upstox API error: {msg}")
        return body

    # ── Market Quote (LTP / Last Price) ──
    def get_spot_price(self, instrument_key: str) -> float:
        """Fetch the last traded price for an index or instrument via /market-quote/ltp."""
        url = f"{UPSTOX_BASE}/market-quote/ltp"
        params = {"instrument_key": instrument_key}
        r = requests.get(url, headers=self.headers, params=params, timeout=10)
        r.raise_for_status()
        data = self._safe_json(r)

        data_body = data.get("data", {})
        # The API nests data under the instrument key
        if instrument_key in data_body:
            return float(data_body[instrument_key]["last_price"])

        # Fallback: try case-insensitive / whitespace-normalised match
        norm_key = instrument_key.lower().replace(" ", "")
        for key, val in data_body.items():
            if key.lower().replace(" ", "") == norm_key:
                return float(val["last_price"])

        # Last resort: first available key
        first_key = next(iter(data_body), None)
        if first_key:
            return float(data_body[first_key]["last_price"])

        raise ValueError(f"Symbol not found in LTP response: {instrument_key}")

    # ── Option Chain Expiry List ──
    def get_expiries(self, instrument_key: str) -> list:
        """Retrieve available expiry dates for an instrument via /option/contract."""
        url = f"{UPSTOX_BASE}/option/contract"
        params = {"instrument_key": instrument_key}
        r = requests.get(url, headers=self.headers, params=params, timeout=10)
        r.raise_for_status()
        data = self._safe_json(r)

        expiries = sorted(set(
            c.get("expiry", "")[:10]
            if isinstance(c.get("expiry"), str) else str(c.get("expiry", ""))[:10]
            for c in data.get("data", [])
        ))
        return [e for e in expiries if e and e != "None"]

    # ── Option Chain Data ──
    def get_option_chain(self, instrument_key: str, expiry_date: str) -> list:
        """Fetch the full option chain for an expiry via /option/chain."""
        url = f"{UPSTOX_BASE}/option/chain"
        params = {"instrument_key": instrument_key, "expiry_date": expiry_date}
        r = requests.get(url, headers=self.headers, params=params, timeout=10)
        r.raise_for_status()
        data = self._safe_json(r)
        return data.get("data", [])

    # ── Historical Candles ──
    def get_historical_candles(self, instrument_key: str, interval: str = "day", days: int = 45) -> pd.DataFrame:
        """Fetch OHLC candles via /historical-candle/{key}/{interval}/{to}/{from}."""
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        encoded_key = urllib.parse.quote(instrument_key, safe="")
        url = f"{UPSTOX_BASE}/historical-candle/{encoded_key}/{interval}/{to_date}/{from_date}"

        r = requests.get(url, headers=self.headers, timeout=10)
        r.raise_for_status()
        data = self._safe_json(r)
        candles = data.get("data", {}).get("candles", [])

        if not candles:
            return pd.DataFrame()

        rows = []
        for c in candles:
            if len(c) >= 6:
                rows.append({
                    "timestamp": c[0],
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": int(c[5]) if len(c) > 5 else 0,
                })
        cdf = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return cdf


# ══════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ══════════════════════════════════════════════

def compute_adx(candles_df: pd.DataFrame, period: int = 14):
    """Compute Average Directional Index (ADX) with +DI / -DI."""
    if candles_df.empty or len(candles_df) < (period * 2 + 2):
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

    plus_di_list, minus_di_list, dx_list = [], [], []

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
    """Compute Black-Scholes price + Greeks for a European option."""
    if T <= 0 or sigma <= 0:
        return {"price": 0, "delta": 0, "gamma": 0, "theta": 0, "vega": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    if option_type == "CE":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    return {
        "price": round(price, 2),
        "delta": round(delta, 3),
        "gamma": round(gamma, 5),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
    }


def compute_max_pain(df_chain: pd.DataFrame, diff: float) -> float:
    """
    Compute the Max Pain strike — the price at which total option buyer losses
    are maximised (i.e., where writers keep the most premium).
    """
    strikes = df_chain["Strike"].values
    pain = {}
    for s in strikes:
        total = 0.0
        for _, row in df_chain.iterrows():
            ce_loss = max(0, s - row["Strike"]) * row["PE OI"]
            pe_loss = max(0, row["Strike"] - s) * row["CE OI"]
            total += ce_loss + pe_loss
        pain[s] = total
    if not pain:
        return 0.0
    return min(pain, key=pain.get)


# ══════════════════════════════════════════════
#  SIDEBAR SETUP
# ══════════════════════════════════════════════

st.sidebar.markdown('<div class="sidebar-header"><b>🔐 Authentication</b></div>', unsafe_allow_html=True)
api_token = st.sidebar.text_input(
    "🔓 Upstox Access Token (Bearer)",
    type="password",
    value="",
    help="Paste the access_token you receive after the OAuth redirect. Tokens expire at midnight IST daily.",
)

st.sidebar.markdown("---")
selected_index_name = st.sidebar.selectbox("🎯 Select Underlying Index", list(INDICES.keys()))

st.sidebar.markdown("---")
st.sidebar.header("🔧 Alpha System Multipliers")
iv_override = st.sidebar.slider(
    "Flat Implied Volatility (%)", min_value=5.0, max_value=80.0, value=15.0, step=0.5
) / 100
risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)", min_value=0.0, max_value=12.0, value=7.0, step=0.1
) / 100
strike_depth = st.sidebar.slider(
    "Strike Range Bracket Around ATM", min_value=3, max_value=15, value=7
)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Enable Auto-Refresh (30s Triggers)", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 120, 30, step=5)

# ══════════════════════════════════════════════
#  CORE ENGINE EVALUATION LOOP
# ══════════════════════════════════════════════

if not api_token:
    st.info(
        "💡 Please enter your **Upstox Access Token** in the sidebar to initialize the engine. "
        "The token is obtained via Upstox OAuth2 flow and expires at midnight IST daily."
    )
else:
    try:
        client = UpstoxClient(token=api_token)
        index_meta = INDICES[selected_index_name]
        diff = index_meta["diff"]

        # Pull underlying execution matrix values
        spot_price = client.get_spot_price(index_meta["key"])
        expiries = client.get_expiries(index_meta["key"])

        if not expiries:
            st.error("No valid active derivative contract windows resolved from the broker API.")
            st.stop()

        selected_expiry = st.sidebar.selectbox("📅 Expiry Window Target", expiries, index=0)

        expiry_dt = datetime.strptime(selected_expiry, "%Y-%m-%d").replace(hour=15, minute=30)
        time_to_expiry_years = (expiry_dt - datetime.now()).total_seconds() / (86400 * 365)
        time_to_expiry_years = max(time_to_expiry_years, 0.0001)

        with st.spinner("Analyzing Index Volatility Structure..."):
            candles_df = client.get_historical_candles(index_meta["key"], interval="day", days=45)
            adx_metrics = compute_adx(candles_df)
            chain_raw = client.get_option_chain(index_meta["key"], selected_expiry)

        if not chain_raw:
            st.warning("Empty option chain returned for the selected expiry. The contract may not be active yet.")
            st.stop()

        atm_strike = round(spot_price / diff) * diff

        # ── Build chain DataFrame ──
        chain_records = []
        for strike_data in chain_raw:
            strike_price = float(strike_data.get("strike_price", 0))
            if abs(strike_price - atm_strike) <= (strike_depth * diff):
                ce = strike_data.get("call_options", {})
                pe = strike_data.get("put_options", {})

                ce_md = ce.get("market_data", {}) if ce else {}
                pe_md = pe.get("market_data", {}) if pe else {}

                ce_oi = ce_md.get("oi", 0)
                pe_oi = pe_md.get("oi", 0)
                ce_ltp = ce_md.get("ltp", 0)
                pe_ltp = pe_md.get("ltp", 0)
                ce_volume = ce_md.get("volume", 0)
                pe_volume = pe_md.get("volume", 0)
                ce_iv = ce_md.get("iv", iv_override * 100)  # Use market IV if available
                pe_iv = pe_md.get("iv", iv_override * 100)

                # Use market IV for Greeks when available, else fall back to manual override
                ce_sigma = (ce_iv / 100) if ce_iv and ce_iv > 0 else iv_override
                pe_sigma = (pe_iv / 100) if pe_iv and pe_iv > 0 else iv_override

                ce_greeks = black_scholes_greeks(spot_price, strike_price, time_to_expiry_years, risk_free_rate, ce_sigma, "CE")
                pe_greeks = black_scholes_greeks(spot_price, strike_price, time_to_expiry_years, risk_free_rate, pe_sigma, "PE")

                chain_records.append({
                    "CE OI": ce_oi,
                    "CE Vol": ce_volume,
                    "CE IV": round(ce_sigma * 100, 1),
                    "CE Delta": ce_greeks["delta"],
                    "CE Gamma": ce_greeks["gamma"],
                    "CE Theta": ce_greeks["theta"],
                    "CE Vega": ce_greeks["vega"],
                    "CE LTP": ce_ltp,
                    "Strike": strike_price,
                    "PE LTP": pe_ltp,
                    "PE Vega": pe_greeks["vega"],
                    "PE Theta": pe_greeks["theta"],
                    "PE Gamma": pe_greeks["gamma"],
                    "PE Delta": pe_greeks["delta"],
                    "PE IV": round(pe_sigma * 100, 1),
                    "PE Vol": pe_volume,
                    "PE OI": pe_oi,
                })

        df_chain = pd.DataFrame(chain_records).sort_values("Strike").reset_index(drop=True)

        # ── Derived metrics ──
        total_ce_oi = df_chain["CE OI"].sum()
        total_pe_oi = df_chain["PE OI"].sum()
        pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0

        max_pain_strike = compute_max_pain(df_chain, diff)

        # Identify resistance (highest CE OI) and support (highest PE OI) strikes
        resistance_strike = df_chain.loc[df_chain["CE OI"].idxmax(), "Strike"] if not df_chain.empty else atm_strike
        support_strike = df_chain.loc[df_chain["PE OI"].idxmax(), "Strike"] if not df_chain.empty else atm_strike

        # ── Sentiment classification ──
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

        # ══════════════════════════════════════════════
        #  TOP KPI METRICS ROW
        # ══════════════════════════════════════════════

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Underlying Spot Price", f"₹ {spot_price:,.2f}")
        m_col2.metric("Calculated ATM Level", f"{atm_strike:,.0f}")
        m_col3.metric("Aggregate PCR (OI)", f"{pcr}")
        m_col4.metric("Max Pain Strike", f"{max_pain_strike:,.0f}")
        if adx_metrics:
            m_col5.metric(
                "ADX (14 Period)",
                f"{adx_metrics['adx']}",
                f"+DI {adx_metrics['plus_di']} / -DI {adx_metrics['minus_di']}",
            )
        else:
            m_col5.metric("ADX (14 Period)", "Insufficient Data")

        # ── Sentiment card ──
        st.markdown(f"""
        <div class="direction-card" style="{card_bg}">
            <div class="score-label">AUTOMATED STRUCTURAL TREND SIGNAL</div>
            <div class="direction-text">{sentiment}</div>
            <div class="sentiment-text">
                Put OI: {total_pe_oi:,.0f} &nbsp;|&nbsp; Call OI: {total_ce_oi:,.0f}
                &nbsp;&nbsp;•&nbsp;&nbsp;
                <span class="sr-badge sr-support">Support: {support_strike:,.0f}</span>
                <span class="sr-badge sr-resist">Resistance: {resistance_strike:,.0f}</span>
                <span class="sr-badge sr-maxpain">Max Pain: {max_pain_strike:,.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ══════════════════════════════════════════════
        #  NORMAL DISTRIBUTION PRICE PREDICTION
        # ══════════════════════════════════════════════

        st.subheader("🎯 Statistical Price Prediction Range (Normal Distribution Model)")

        std_dev_price = spot_price * iv_override * np.sqrt(time_to_expiry_years)
        lower_1sigma = spot_price - std_dev_price
        upper_1sigma = spot_price + std_dev_price
        lower_2sigma = spot_price - (2 * std_dev_price)
        upper_2sigma = spot_price + (2 * std_dev_price)

        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("1σ Low Bound (68.2%)", f"₹ {lower_1sigma:,.2f}")
        p_col2.metric("🎯 Median Expected Spot", f"₹ {spot_price:,.2f}")
        p_col3.metric("1σ High Bound (68.2%)", f"₹ {upper_1sigma:,.2f}")

        ic_sell_put = round(lower_1sigma / diff) * diff
        ic_sell_call = round(upper_1sigma / diff) * diff
        ic_buy_put = ic_sell_put - diff
        ic_buy_call = ic_sell_call + diff

        fig_bell, ax_bell = plt.subplots(figsize=(12, 4.5))
        x_axis = np.linspace(spot_price - 3.5 * std_dev_price, spot_price + 3.5 * std_dev_price, 500)
        y_axis = norm.pdf(x_axis, spot_price, std_dev_price)

        ax_bell.plot(x_axis, y_axis, color="#0f172a", linewidth=2, label="Probability Density Function")
        ax_bell.fill_between(
            x_axis, y_axis,
            where=(x_axis >= lower_1sigma) & (x_axis <= upper_1sigma),
            color="#38bdf8", alpha=0.35, label="68.2% Confidence Zone (1σ)",
        )
        ax_bell.fill_between(
            x_axis, y_axis,
            where=((x_axis >= lower_2sigma) & (x_axis < lower_1sigma)) |
                  ((x_axis > upper_1sigma) & (x_axis <= upper_2sigma)),
            color="#0284c7", alpha=0.15, label="95.4% Confidence Zone (2σ)",
        )

        ax_bell.axvline(spot_price, color="#0f172a", linestyle="-", linewidth=1.5,
                        label=f"Current Spot ({spot_price:,.1f})")
        ax_bell.axvline(max_pain_strike, color="#7c3aed", linestyle="-.", linewidth=1.5,
                        label=f"Max Pain ({max_pain_strike:,.0f})")
        ax_bell.axvline(ic_sell_put, color="#e11d48", linestyle=":", linewidth=2,
                        label=f"IC Sell Floor ({ic_sell_put})")
        ax_bell.axvline(ic_sell_call, color="#16a34a", linestyle=":", linewidth=2,
                        label=f"IC Sell Cap ({ic_sell_call})")

        ax_bell.set_title(
            f"Expiry Statistical Forecast — {selected_index_name} ({selected_expiry})",
            fontsize=11, fontweight="bold",
        )
        ax_bell.set_xlabel("Predicted Index Settlement Price", fontsize=9)
        ax_bell.set_ylabel("Probability Density", fontsize=9)
        ax_bell.legend(loc="upper right", fontsize=8)
        ax_bell.grid(True, linestyle=":", alpha=0.4)
        st.pyplot(fig_bell)
        plt.close(fig_bell)

        # ══════════════════════════════════════════════
        #  STRATEGY PLAYBOOK
        # ══════════════════════════════════════════════

        st.subheader("🛡️ Automated Statistical Strategy Playbook")

        adx_val = adx_metrics["adx"] if adx_metrics else 15
        if adx_val > 25:
            suggested_strategy = "Switch to Debit Spreads / Long Options (Trending Market)"
        elif iv_override > 0.22:
            suggested_strategy = "Iron Butterfly"
        else:
            suggested_strategy = "Iron Condor"

        st.info(
            f"📊 **Matrix Recommendation**: ADX = **{adx_val}**, IV = **{iv_override*100:.1f}%** "
            f"→ Suggested setup: **{suggested_strategy}**"
        )

        selected_strategy_view = st.selectbox(
            "⚡ Select Active Strategy Playbook View",
            ["Iron Condor", "Short Straddle", "Iron Butterfly"],
        )

        def get_prices(strike):
            """Look up live CE & PE LTP from the chain for a given strike."""
            match = df_chain[df_chain["Strike"] == strike]
            if not match.empty:
                return float(match.iloc[0]["CE LTP"]), float(match.iloc[0]["PE LTP"])
            return 0.0, 0.0

        if selected_strategy_view == "Iron Condor":
            c_sell_ce, _ = get_prices(ic_sell_call)
            _, p_sell_pe = get_prices(ic_sell_put)
            c_buy_ce, _ = get_prices(ic_buy_call)
            _, p_buy_pe = get_prices(ic_buy_put)

            net_credit = (c_sell_ce + p_sell_pe) - (c_buy_ce + p_buy_pe)
            net_credit = max(net_credit, 0.0)
            max_risk = diff - net_credit if net_credit > 0 else diff

            st.markdown(f"""
            <div class="playbook-card">
                <div class="playbook-title">📊 Iron Condor (Risk-Defined)</div>
                <strong>Legs:</strong><br>
                • Buy 1x <b>{ic_buy_put} PE</b> @ ₹{p_buy_pe:.2f}<br>
                • Sell 1x <b>{ic_sell_put} PE</b> @ ₹{p_sell_pe:.2f}<br>
                • Sell 1x <b>{ic_sell_call} CE</b> @ ₹{c_sell_ce:.2f}<br>
                • Buy 1x <b>{ic_buy_call} CE</b> @ ₹{c_buy_ce:.2f}<br><br>
                <span style='color: #16a34a; font-size: 16px; font-weight: bold;'>
                    💰 Net Credit: ₹{net_credit:,.2f} / lot
                </span>
                &nbsp;&nbsp;
                <span style='color: #dc2626; font-size: 13px;'>
                    Max Risk: ₹{max_risk:,.2f} / lot
                </span>
            </div>
            """, unsafe_allow_html=True)

        elif selected_strategy_view == "Short Straddle":
            c_sell_ce, p_sell_pe = get_prices(atm_strike)
            max_profit = c_sell_ce + p_sell_pe
            upper_be = atm_strike + max_profit
            lower_be = atm_strike - max_profit

            st.markdown(f"""
            <div class="playbook-card">
                <div class="playbook-title">🔥 Short Straddle (Max Premium Harvest)</div>
                <strong>Legs:</strong><br>
                • Sell 1x <b>{atm_strike} CE</b> @ ₹{c_sell_ce:.2f}<br>
                • Sell 1x <b>{atm_strike} PE</b> @ ₹{p_sell_pe:.2f}<br><br>
                <span style='color: #16a34a; font-size: 16px; font-weight: bold;'>
                    💰 Net Credit: ₹{max_profit:,.2f} / lot
                </span><br>
                Breakevens: ₹{lower_be:,.0f} – ₹{upper_be:,.0f}<br>
                <em>⚠️ Unlimited risk. Use trailing stop losses at 2σ boundaries
                (₹{lower_2sigma:,.0f} / ₹{upper_2sigma:,.0f}).</em>
            </div>
            """, unsafe_allow_html=True)

        elif selected_strategy_view == "Iron Butterfly":
            c_sell_ce, p_sell_pe = get_prices(atm_strike)
            c_buy_ce, _ = get_prices(ic_buy_call)
            _, p_buy_pe = get_prices(ic_buy_put)

            net_credit = (c_sell_ce + p_sell_pe) - (c_buy_ce + p_buy_pe)
            net_credit = max(net_credit, 0.0)

            st.markdown(f"""
            <div class="playbook-card">
                <div class="playbook-title">🦋 Iron Butterfly (Volatility Crush)</div>
                <strong>Legs:</strong><br>
                • Buy 1x <b>{ic_buy_put} PE</b> @ ₹{p_buy_pe:.2f}<br>
                • Sell 1x <b>{atm_strike} PE</b> @ ₹{p_sell_pe:.2f}<br>
                • Sell 1x <b>{atm_strike} CE</b> @ ₹{c_sell_ce:.2f}<br>
                • Buy 1x <b>{ic_buy_call} CE</b> @ ₹{c_buy_ce:.2f}<br><br>
                <span style='color: #16a34a; font-size: 16px; font-weight: bold;'>
                    💰 Net Credit: ₹{net_credit:,.2f} / lot
                </span>
            </div>
            """, unsafe_allow_html=True)

        # ══════════════════════════════════════════════
        #  OPTIONS CHAIN TABLE
        # ══════════════════════════════════════════════

        st.subheader("📊 Live Options Chain — Greeks & Market Data")

        def color_strikes(row):
            """Highlight ITM calls green, ITM puts red, ATM bold."""
            val = row["Strike"]
            base = [""] * len(row)
            if val == atm_strike:
                base = ["background-color: #fef9c3; font-weight: bold;"] * len(row)
            elif val < spot_price:
                base = ["background-color: #f0fdf4;"] * len(row)
            elif val > spot_price:
                base = ["background-color: #fef2f2;"] * len(row)
            return base

        styled_df = df_chain.style.apply(color_strikes, axis=1).format({
            "CE OI": "{:,.0f}", "CE Vol": "{:,.0f}", "CE IV": "{:.1f}%",
            "CE Delta": "{:.3f}", "CE Gamma": "{:.5f}", "CE Theta": "{:.2f}", "CE Vega": "{:.2f}",
            "CE LTP": "₹{:.2f}",
            "Strike": "{:,.0f}",
            "PE LTP": "₹{:.2f}",
            "PE Vega": "{:.2f}", "PE Theta": "{:.2f}", "PE Gamma": "{:.5f}", "PE Delta": "{:.3f}",
            "PE IV": "{:.1f}%", "PE Vol": "{:,.0f}", "PE OI": "{:,.0f}",
        })
        st.dataframe(styled_df, use_container_width=True, height=420)

        # ══════════════════════════════════════════════
        #  CHART 1: OPEN INTEREST DISTRIBUTION
        # ══════════════════════════════════════════════

        st.subheader("📈 Open Interest Distribution Profile")

        fig_oi, ax_oi = plt.subplots(figsize=(12, 4.5))
        width = diff * 0.35
        strikes = df_chain["Strike"].values

        ax_oi.bar(strikes - width / 2, df_chain["CE OI"], width,
                  label="Call OI", color="#ef4444", alpha=0.85, edgecolor="#b91c1c", linewidth=0.5)
        ax_oi.bar(strikes + width / 2, df_chain["PE OI"], width,
                  label="Put OI", color="#22c55e", alpha=0.85, edgecolor="#15803d", linewidth=0.5)

        ax_oi.axvline(atm_strike, color="#0f172a", linestyle="--", linewidth=1.2, label=f"ATM ({atm_strike:,.0f})")
        ax_oi.axvline(max_pain_strike, color="#7c3aed", linestyle="-.", linewidth=1.2,
                      label=f"Max Pain ({max_pain_strike:,.0f})")

        ax_oi.set_title(f"OI Distribution — {selected_index_name} ({selected_expiry})", fontsize=11, fontweight="bold")
        ax_oi.set_xlabel("Strike Price", fontsize=9)
        ax_oi.set_ylabel("Open Interest (Contracts)", fontsize=9)
        ax_oi.legend(loc="upper right", fontsize=8)
        ax_oi.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax_oi.set_xticks(strikes)
        ax_oi.set_xticklabels([f"{int(s)}" for s in strikes], rotation=45, fontsize=7)
        st.pyplot(fig_oi)
        plt.close(fig_oi)

        # ══════════════════════════════════════════════
        #  CHART 2: DELTA SKEW CURVE
        # ══════════════════════════════════════════════

        st.subheader("📉 Delta Skew Profile (CE vs PE)")

        fig_delta, ax_delta = plt.subplots(figsize=(12, 3.5))
        ax_delta.plot(df_chain["Strike"], df_chain["CE Delta"], marker="o", markersize=4,
                      color="#0284c7", linewidth=1.8, label="CE Delta")
        ax_delta.plot(df_chain["Strike"], df_chain["PE Delta"], marker="s", markersize=4,
                      color="#e11d48", linewidth=1.8, label="PE Delta")
        ax_delta.axhline(0, color="#94a3b8", linestyle="-", linewidth=0.8)
        ax_delta.axvline(atm_strike, color="#0f172a", linestyle="--", linewidth=1, alpha=0.6,
                         label=f"ATM ({atm_strike:,.0f})")

        ax_delta.set_title("Delta Skew Across Strikes", fontsize=11, fontweight="bold")
        ax_delta.set_xlabel("Strike Price", fontsize=9)
        ax_delta.set_ylabel("Delta", fontsize=9)
        ax_delta.legend(loc="center right", fontsize=8)
        ax_delta.grid(True, linestyle=":", alpha=0.4)
        st.pyplot(fig_delta)
        plt.close(fig_delta)

        # ══════════════════════════════════════════════
        #  CHART 3: IV SMILE (if market IV available)
        # ══════════════════════════════════════════════

        st.subheader("😊 Implied Volatility Smile")

        fig_iv, ax_iv = plt.subplots(figsize=(12, 3.5))
        ax_iv.plot(df_chain["Strike"], df_chain["CE IV"], marker="^", markersize=4,
                   color="#0284c7", linewidth=1.8, label="CE IV")
        ax_iv.plot(df_chain["Strike"], df_chain["PE IV"], marker="v", markersize=4,
                   color="#e11d48", linewidth=1.8, label="PE IV")
        ax_iv.axvline(atm_strike, color="#0f172a", linestyle="--", linewidth=1, alpha=0.6)

        ax_iv.set_title("IV Smile / Skew Across Strikes", fontsize=11, fontweight="bold")
        ax_iv.set_xlabel("Strike Price", fontsize=9)
        ax_iv.set_ylabel("Implied Volatility (%)", fontsize=9)
        ax_iv.legend(loc="upper right", fontsize=8)
        ax_iv.grid(True, linestyle=":", alpha=0.4)
        st.pyplot(fig_iv)
        plt.close(fig_iv)

        # ══════════════════════════════════════════════
        #  CHART 4: OI-WEIGHTED PUT/CALL PRESSURE
        # ══════════════════════════════════════════════

        st.subheader("⚖️ Cumulative OI Pressure (Call vs Put)")

        fig_cum, ax_cum = plt.subplots(figsize=(12, 3.5))
        df_sorted = df_chain.sort_values("Strike")
        ax_cum.fill_between(df_sorted["Strike"], df_sorted["CE OI"].cumsum(), alpha=0.3, color="#ef4444", label="Cumul. Call OI")
        ax_cum.fill_between(df_sorted["Strike"], df_sorted["PE OI"].cumsum(), alpha=0.3, color="#22c55e", label="Cumul. Put OI")
        ax_cum.plot(df_sorted["Strike"], df_sorted["CE OI"].cumsum(), color="#ef4444", linewidth=1.5)
        ax_cum.plot(df_sorted["Strike"], df_sorted["PE OI"].cumsum(), color="#22c55e", linewidth=1.5)
        ax_cum.axvline(atm_strike, color="#0f172a", linestyle="--", linewidth=1, alpha=0.6)

        ax_cum.set_title("Cumulative OI Build-up", fontsize=11, fontweight="bold")
        ax_cum.set_xlabel("Strike Price", fontsize=9)
        ax_cum.set_ylabel("Cumulative OI", fontsize=9)
        ax_cum.legend(fontsize=8)
        ax_cum.grid(True, linestyle=":", alpha=0.4)
        st.pyplot(fig_cum)
        plt.close(fig_cum)

        # ══════════════════════════════════════════════
        #  FOOTER: LAST REFRESH + DATA FRESHNESS
        # ══════════════════════════════════════════════

        now_ist = datetime.now() + timedelta(hours=0)  # Server is already IST
        st.markdown("---")
        fc1, fc2, fc3 = st.columns(3)
        fc1.caption(f"🕐 Last Refresh: {now_ist.strftime('%H:%M:%S IST')}")
        fc2.caption(f"📅 Expiry: {selected_expiry} | T = {max(time_to_expiry_years * 365, 0):.2f} days")
        fc3.caption(f"📊 Chain Strikes Loaded: {len(df_chain)}")

        # ══════════════════════════════════════════════
        #  AUTO-REFRESH TRIGGER
        # ══════════════════════════════════════════════

        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

    except ValueError as ve:
        st.error(f"🚫 **Data Error**: {ve}")
    except requests.exceptions.HTTPError as he:
        status = he.response.status_code if he.response is not None else "N/A"
        st.error(
            f"🚫 **HTTP {status}**: The Upstox API rejected the request. "
            f"If status is 401/403, your access token has likely expired (they reset at midnight IST). "
            f"Re-authenticate via the Upstox OAuth flow to get a fresh token."
        )
    except requests.exceptions.ConnectionError:
        st.error("🚫 **Connection Error**: Could not reach the Upstox API. Check your network or try again.")
    except requests.exceptions.Timeout:
        st.error("🚫 **Timeout**: The Upstox API took too long to respond. Try again in a few seconds.")
    except Exception as e:
        st.error(f"🚫 **Unexpected Error**: {type(e).__name__}: {e}")
        st.exception(e)
