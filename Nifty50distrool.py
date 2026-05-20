import streamlit as st
import pandas as pd
import numpy as np
import requests
import urllib.parse
import plotly.graph_objects as go
from scipy.stats import norm

# --- DASHBOARD LAYOUT CONFIGURATION ---
st.set_page_config(page_title="Multi-Asset Probability Dashboard", layout="wide")

st.title("📊 Multi-Asset Probability & Price Prediction Tool")
st.markdown("Select an asset across Nifty or BSE universes to run live mathematical volatility modeling.")

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

# Dropdown menu containing your custom structured index catalogs
asset_type = st.sidebar.selectbox(
    "Select Universe", 
    ["Indices", "Nifty 50 Universe", "Nifty 500 Universe", "BSE 500 Universe", "BSE Stocks (Live Search)"]
)

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

# --- SEGMENT 2: NIFTY 50 UNIVERSE ---
elif asset_type == "Nifty 50 Universe":
    nifty_50_str = "RELIANCE,TCS,HDFCBANK,ICICIBANK,INFY,BHARTIARTL,HINDUNILVR,ITC,SBIN,LTIM,ADANIENT,ADANIPORTS,ASIANPAINT,AXISBANK,BAJAJ-AUTO,BAJFINANCE,BAJAJFINSV,BPCL,BRITANNIA,CIPLA,COALINDIA,DIVISLAB,DRREDDY,EICHERMOT,GRASIM,HCLTECH,HEROMOTOCO,HINDALCO,INDUSINDBK,JSWSTEEL,KOTAKBANK,LT,M&M,MARUTI,NESTLEIND,NTPC,ONGC,POWERGRID,SBILIFE,SUNPHARMA,TATACONSUM,TATAMOTORS,TATASTEEL,TECHM,TITAN,ULTRACEMCO,UPL,WIPRO"
    tickers_list = sorted([t.strip() for t in nifty_50_str.split(",")])
    
    selected_ticker = st.sidebar.selectbox("Select NSE Ticker", tickers_list)
    selected_asset_name = f"{selected_ticker} (NSE)"
    target_instrument_key = f"NSE_EQ|{selected_ticker}"

# --- SEGMENT 3: NIFTY 500 UNIVERSE ---
elif asset_type == "Nifty 500 Universe":
    nifty_500_str = "360ONE,3MINDIA,ABB,ACC,AIAENG,APLAPOLLO,AUBANK,AETHER,AFFLE,AJANTPHARM,APLLTD,ALKEM,ALKYLAMINE,ALLCARGO,ALOKINDS,AMBER,AMBUJACEM,ANANTRAJ,ANGELONE,APARINDS,APOLLOHOSP,APOLLOTYRE,APTUS,ARE&M,ASAHIINDIA,ASHOKLEY,ASIANPAINT,ASTERDM,ASTRAL,ATUL,AUROPHARMA,AVANTIFEED,DMART,BEML,BLS,BSE,BALAMINES,BALKRISIND,BALRAMCHIN,BANDHANBNK,BANKBARODA,BANKINDIA,MAHABANK,BATAINDIA,BAYERCROP,BERGEPAINT,BDL,BEL,BHARATFORG,BHEL,BIOCON,BIRLACORPN,BSOFT,BLUEDART,BORORENEW,BOSCHLTD,CAMPUS,CESC,CGPOWER,CIEINDIA,CRISIL,CSBBANK,CANFINHOME,CANBK,CAPLIPOINT,CGCL,CARBORUNIV,CASTROLIND,CEATLTD,CENTRALBK,CDSL,CENTURYPLY,CENTURYTEX,CHAMBLFERT,CHALET,CHOLAFIN,CHOLAHLDNG,CUB,CIPLA,CLEAN,COALINDIA,COCHINSHIP,COFORGE,COLPAL,CONCOR,COROMANDEL,CRAFTSMAN,CREDITACC,CROMPTON,CUMMINSIND,CYIENT,DCMSHRIRAM,DLF,DABUR,DALBHARAT,DEEPAKFERT,DEEPAKNTR,DELHIVERY,DEVYANI,DIXON,DRREDDY,EIDPARRY,EIHOTEL,EPL,EASEMYTRIP,EICHERMOT,ELECON,EMAMILTD,ENDURANCE,ENGINERSIN,ERIS,ESCORTS,EXIDEIND,FDC,FSNKYS,FEDERALBNK,FACT,FINEORG,FINCABLES,FINPIPE,FSL,FORTIS,GRINFRA,GAIL,GLS,GMRINFRA,GEPIL,GHCL,GICRE,GIPCL,GLAXO,GLENMARK,GODREJAGRO,GODREJCP,GODREJPROP,GRANULES,GRAPHITE,GRASIM,GESHIP,GREAVESCOT,GRINDWELL,GUJALKALI,GUJGASLTD,GMDCLTD,GNFC,GPPL,GSFC,GSPL,HCLTECH,HDFCAMC,HDFCBANK,HDFCLIFE,HFCL,HLEGLAS,HAPPSTMNDS,HAVELLS,HEG,HEROMOTOCO,HINDALCO,HCOPPER,HINDPETRO,HINDUNILVR,HINDZINC,HONAUT,HUDCO,ICICIBANK,ICICIGI,ICICIPRULI,ISEC,IDBI,IDFCFIRSTB,IDFC,IIFL,IRB,IRCON,ITC,ITI,ITDCEM,INDIACEM,IBREALEST,INDIAMART,INDIANB,IEX,INDHOTEL,IOC,IRCTC,IRFC,INDIGOPNTS,IGL,INDUSINDBK,INDUSTOWER,INFIBEAM,INFY,INOXWIND,INTELLECT,INDIGO,IPCALAB,JBCHEPHARM,JKCEMENT,JKLACEMENT,JKPAPER,JMFINANCIL,JSWENERGY,JSWSTEEL,JSWINFRA,JAMNAAUTO,JSL,JINDALSTEL,JINDWORLD,JUBLFOOD,JUBLPHARMA,JUBLINGREA,JUSTDIAL,JYOTHYLAB,KALYANKJIL,KEI,KNRCON,KPITTECH,KRBL,KSB,KAJARIACER,KANSNEROL,KARURVYSYA,KEC,KENNAMET,KIMS,KIRLOSENG,KIRLPNU,KOLTEPATIL,KOTAKBANK,L&TFH,LTTS,LICHSGFIN,LICI,LAURUSLABS,LXCHEM,LEMONTREE,LINDEINDIA,LUPIN,LUXIND,MASFIN,MRF,MTARTECH,MTNL,MGL,MAHSEAMLES,M&MFIN,M&M,MANAPPURAM,MARICO,MARUTI,MASTEK,MEDPLUS,METROPOLIS,MFSL,MINDACORP,MSUMI,MOTILALOFS,MPHASIS,MCX,MUTHOOTFIN,NHPC,NLCINDIA,NMDC,NOCIL,NTPC,NATIONALUM,NAVINFLUOR,NAZARA,NEOGEN,NESCO,NESTLEIND,NETWORK18,NIPPONLIIF,OBEROIRLTY,ONGC,OIL,OLECTRA,PAYTM,OFSS,ORIENTELEC,POLICYBZR,PCBL,PIIND,PNBHOUSING,PNCINFRA,PVRINOX,PAGEIND,PATANJALI,PEL,PFC,POWERGRID,PRESTIGE,PRINCEPIPE,PRAJIND,PRIVISCL,PNB,QUESS,RBLBANK,RECLTD,RHIM,RITES,RADICO,RAIN,RAINBOW,RAJESHEXPO,RALLIS,RAMCOCEM,RATNAMANI,RAYMOND,REDINGTON,RELAXO,RELIANCE,RELIGARE,RVNL,SJVN,SKFINDIA,SRF,SANOFI,SAPPHIRE,SAREGAMA,SASTASUNDR,SBICARD,SBILIFE,SCHAEFFLER,SHOPERSTOP,SHREECEM,SHRIRAMFIN,SIEMENS,SOBHA,SOLARINDS,SONACOMS,SONATSOFTW,STARHEALTH,SBI,SAIL,SUNPHARMA,SUNTV,SUNDARMFIN,SUNDRMFAST,SUNTECK,SUPRAJIT,SUPREMEIND,SUZLON,SWANENERGY,SYNGENE,TARC,TCIEXP,TTKPRESTIG,TV18BRDCST,TVSMOTOR,TARSONS,TATACONSUM,TATACOMM,TATAELXSI,TATAMOTORS,TATAPOWER,TATASTEEL,TATAINVEST,TATATECH,TECHM,TEJASNET,TIMKEN,TITAN,TORNTPHARM,TORNTPOWER,TREND,TRIDENT,TRIVENI,UCOBANK,UBL,UDEV,UNIONBANK,UPL,UTIAMC,VGUARD,VMART,VIPIND,VAIBHAVGBL,VAKRANGEE,VARROC,VBL,VEDL,VINATIORG,VOLTAS,WELCORP,WELSPUNLIV,WESTLIFE,WHIRLPOOL,WIPRO,YESBANK,ZFCVINDIA,ZEEL,ZENSARTECH,ZOMATO,ZYDUSLIFE"
    tickers_list = sorted([t.strip() for t in nifty_500_str.split(",")])
    
    selected_ticker = st.sidebar.selectbox("Select NSE 500 Ticker", tickers_list)
    selected_asset_name = f"{selected_ticker} (NSE)"
    target_instrument_key = f"NSE_EQ|{selected_ticker}"

# --- SEGMENT 4: BSE 500 UNIVERSE ---
elif asset_type == "BSE 500 Universe":
    bse_500_str = "GRSE,ETERNAL,RELIANCE,BANDHANBNK,VEDL,MAZDOCK,HDFCBANK,SUNPHARMA,COCHINSHIP,CEATLTD,M&M,SBIN,ADANIPOWER,MARUTI,GROWW,COALINDIA,ICICIBANK,BSE,DATAPATTNS,EMMVEE,ONGC,TENNIND,CHENNPETRO,BHARTIARTL,NETWEB,INFY,MCX,ITC,DIXON,SCI,ADANIENT,RECLTD,IDEA,SUZLON,TATASTEEL,AXISBANK,RBLBANK,LT,GMDCLTD,JIOFIN,STARHEALTH,CROMPTON,DRREDDY,INDIGO,OFSS,HCLTECH,TCS,WAAREEENER,SHRIRAMFIN,PFC,GODFRYPHLP,ATGL,BAJFINANCE,TMPV,GESHIP,JPPOWER,VBL,COHANCE,ADANIGREEN,BPCL,SWIGGY,POWERINDIA,INDUSTOWER,ADANIPORTS,ENRIN,HSCL,SWANCORP,EMCURE,TECHM,LODHA,NESTLEIND,SAIL,HINDZINC,FORCEMOT,BHEL,PERSISTENT,NATIONALUM,SAMMAANCAP,KAYNES,BHARATFORG,ULTRACEMCO,INDUSINDBK,PIRAMALFIN,TATAPOWER,ADANIENSOL,WELCORP,TVSMOTOR,EICHERMOT,HINDUNILVR,HINDALCO,NMDC,BAJAJ-AUTO,BEL,TATACHEM,PAYTM,JSWSTEEL,CANBK,GVT&D,ASHOKLEY,NHPC,TRENT,OIL,HDFCLIFE,HAL,WIPRO,OLAELEC,UNIONBANK,BRITANNIA,MAHABANK,TMCV,ABCAPITAL,NTPC,AWL,HAVELLS,POWERGRID,HINDCOPPER,HEROMOTOCO,YESBANK,SONACOMS,HFCL,NAUKRI,ABB,CDSL,JAINREC,KOTAKBANK,IDFCFIRSTB,AUROPHARMA,POLYCAB,KEI,BDL,TITAN,FEDERALBNK,RVNL,ATHERENERG,APOLLOSP,NAVINFLUOR,SAPPHIRE,INDIANB,JKTYRE,TARIL,MAXHEALTH,COFORGE,IGL,EXIDEIND,JSWENERGY,PNB,GRASIM,MOTHERSON,FIVESTAR,RPOWER,HDFCAMC,POLICYBZR,IIFL,GLENMARK,CONCORDBIO,CGPOWER,LAURUSLABS,MRF,TEJASNET,MRPL,BLUESTARCO,ASTRAL,HYUNDAI,GAIL,PPLPHARMA,CUMMINSIND,APARINDS,IOC,GODREJPROP,MUTHOOTFIN,DLF,BANKBARODA,SBILIFE,HINDPETRO,SBICARD,ONESOURCE,ASIANPAINT,MSUMI,TATAELXSI,KALYANKJIL,LLOYDSME,ANGELONE,SUPREMEIND,J&KBANK,NLCINDIA,MOTILALOFS,CANHLIFE,LUPIN,M&MFIN,PNBHOUSING,JINDALSTEL,AMBER,TATACONSUM,SOLARINDS,LENSKART,OLECTRA,BAJAJFINSV,NTPCGREEN,KPITTECH,INDHOTEL,BOSCHLTD,PETRONET,JUBLFOOD,RKFORGE,REDINGTON,GMRAIRPORT,SRF,RRKABEL,AUBANK,ABSLAMC,DIVISLAB,UPL,UNOMINDA,NAM-INDIA,JINDALSAW,HBLENGINE,CGCL,BANKINDIA,JYOTIOTCNC,ZEEL,IRFC,VOLTAS,MPHASIS,DMART,ZENTEC,MANAPPURAM,PGEL,SHYAMMETL,IREDA,LTM,CIPLA,CONCOR,SYRMA,DALBHARAT,LICHSGFIN,IEX,LTF,PIIND,PHOENIXLTD,HUDCO,HEG,CHOLAFIN,GRAPHITE,DEVYANI,GPIL,INOXWIND,KIRLOSENG,AARTIIND,UNITDSPR,ENGINERSIN,LICI,PWL,NCC,APOLLOTYRE,ACUTAAS,ANANDRATHI,KIMS,LGEINDIA,ITCHOTELS,SIEMENS,VMM,RADICO,ANANTRAJ,POONAWALLA,GRANULES,TORNTPHARM,NBCC,INDIACEM,COLPAL,AMBUJACEM,PREMIERENE,IFCI,CUB,BALRAMCHIN,TORNTPOWER,ZYDUSLIFE,TATACAP,360ONE,IDBI,BIOCON,IRCTC,ARE&M,MEESHO,BAJAJHFL,PCBL,CAMS,FORTIS,TRITURBINE,BEML,AFFLE,PARADEEP,ICICIGI,DELHIVERY,MANKIND,INTELLECT,APLAPOLLO,CESC,ELGIEQUIP,IRCON,JWL,WOCKPHARMA,SJVN,NATCOPHARM,JBMA,OBEROIRLTY,KEC,TITAGARH,NYKAA,DEEPAKFERT,KARURVYSYA,JBCHEPHARM,DEEPAKNTR,MFSL,TIINDIA,ABFRL,ICICIAMC,CEMPRO,SAILIFE,KFINTECH,MARICO,PAGEIND,TATATECH,GILLETTE,ZENSARTECH,JSWINFRA,LEMONTREE,BALKRISIND,CHOICEIN,CARTRADE,PRESTIGE,THELEELA,NSLNISP,RAILTEL,FACT,NEWGEN,GLAXO,GODREJCP,LINDEINDIA,TATAINVEST,USHAMART,TATACOMM,IKS,CASTROLIND,GRAVITA,CYIENT,BELRISE,COROMANDEL,ASTERDM,WHIRLPOOL,JSL,PIDILITIND,CLEAN,PATANJALI,LTFOODS,SARDAEN,ACMESOLAR,THERMAX,DABUR,FINCABLES,NH,URBANCO,ABREL,GALLANTT,LALPATHLAB,FSL,SAGILITY,ICICIPRULI,HOMEFIRST,SONATSOFTW,NEULANDLAB,TRIDENT,PVRINOX,SYNGENE,IOB,CPPLUS,SIGNATURE,ALKEM,CCL,ESCORTS,FLUOROCHEM,ELECON,SCHAEFFLER,ATUL,GODREJIND,IRB,LTTS,FIRSTCRY,ECLERX,ENDURANCE,SUNTV,SOBHA,ABBOTINDIA,APTUS,VTL,JMFINANCIL,SHREECEM,BSOFT,ITI,KAJARIACER,CRAFTSMAN,CREDITACC,CHAMBLFERT,TECHNOE,CHOLAHLDNG,SUNDARMFIN,AFCONS,CARBORUNIV,BHARTIHEXA,ACC,ANTHEM,SCHNEIDER,BAJAJHLDNG,PINELABS,AEGISLOG,MINDACORP,IPCALAB,CANFINHOME,CENTRALBK,NUVAMA,BLS,NIVABUPA,UCOBANK,NAVA,WELSPUNLIV,AJANTPHARM,GICRE,MEDANTA,JUBLPHARMA,3MINDIA,LATENTVIEW,GABRIEL,TTML,GODIGIT,EMAMILTD,RAINBOW,JKCEMENT,INDGN,ACE,HDBFS,INDIAMART,ABDL,BLUEJET,POLYMED,ZYDUSWELL,CRISIL,KPRMILL,AEGISVOPAK,HEXT,GSPL,MMTC,MGL,AADHARHFC,UBL,ASAHIINDIA,BATAINDIA,EIDPARRY,NUVOCO,DOMS,UTIAMC,NIACL,HONASA,BIKAJI,RAMCOCEM,ZFCVINDIA,ANURAS,AIIL,JSWCEMENT,IGIL,HONAUT,SBFC,RITES,CAPLIPOINT,BRIGADE,SPLPETRO,ERIS,PTCIL,MAPMYINDIA,BAYERCROP,AAVAS,TEGA,SAREGAMA,TBOTEK,VIJAYA,DCMSHRIRAM,AIAENG,GLAND,TIMKEN,JUBLINGREA,CHALET,BERGEPAINT,BBTC,EIHOTEL,KPIL,SUMICHEM,ABLBL,BLUEDART,PFIZER,RHIM,JSWDULUX,TRAVELFOOD"
    tickers_list = sorted([t.strip() for t in bse_500_str.split(",")])
    
    selected_ticker = st.sidebar.selectbox("Select BSE Ticker", tickers_list)
    selected_asset_name = f"{selected_ticker} (BSE)"
    target_instrument_key = f"BSE_EQ|{selected_ticker}"

# --- SEGMENT 5: LIVE KEYWORD LOOKUP ---
else:
    bse_search = st.sidebar.text_input("Type Custom Keyword Search", value="Reliance")
    if upstox_token and bse_search:
        try:
            search_url = f"https://api.upstox.com/v2/instruments/search?keyword={bse_search}"
            search_headers = {'Accept': 'application/json', 'Authorization': f'Bearer {upstox_token}'}
            search_res = requests.get(search_url, headers=search_headers)
            if search_res.status_code == 200:
                search_data = search_res.json().get('data', [])
                bse_results = [item for item in search_data if item.get('instrument_type') == 'EQUITY']
                if bse_results:
                    bse_map = {f"{item['name']} ({item['trading_symbol']}) - {item['exchange']}": item['instrument_key'] for item in bse_results[:15]}
                    selected_asset_name = st.sidebar.selectbox("Matching Results", list(bse_map.keys()))
                    target_instrument_key = bse_map[selected_asset_name]
        except Exception:
            pass
            
    if not target_instrument_key:
        selected_asset_name = "RELIANCE (NSE Falling Back)"
        target_instrument_key = "NSE_EQ|RELIANCE"

# Horizon mapping configuration parameter
time_horizon = st.sidebar.selectbox("Select Prediction Horizon", ["1 Week", "1 Month", "3 Months", "1 Year"])
dte_mapping = {"1 Week": 7, "1 Month": 30, "3 Months": 90, "1 Year": 365}
days_to_target = dte_mapping[time_horizon]

# --- LIVE MARKET DATA ENGINE ---
def fetch_upstox_live_data(token, instrument_key):
    """
    Queries Upstox v2 Market Quote Endpoint for the chosen asset key
    and safely parses active prices and volatility metrics.
    """
    # URL-encode the instrument key to turn '|' into '%7C'
    encoded_key = urllib.parse.quote(instrument_key)
    url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_key}"
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        json_data = response.json()
        response_key = instrument_key.replace("|", ":")
        
        if 'data' in json_data:
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
                available_keys = list(json_data['data'].keys())
                raise KeyError(f"Target '{response_key}' missing. API returned keys: {available_keys}")
        else:
            raise KeyError("Malformed API response structure: 'data' property missing.")
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

st.plotly_chart(fig, width="stretch")

# --- TRANSLATION SUMMARY PANEL ---
st.markdown("### 📋 Investor Insight Summary")
st.info(f"""
*   📊 **Probability Target Ranges:** Based on current calculations, there is a **68.2% mathematical probability** that **{selected_asset_name}** will trade within the range of **₹{lower_1sig:,.2f}** and **₹{upper_1sig:,.2f}** over the next {time_horizon}.
""")
    
