import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from fpdf import FPDF

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Gorostiaga Research | Terminal Pro", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 32px; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 16px; }
    div[data-testid="stMetric"] { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE DATOS (NUEVA LÓGICA SIN REQUESTS_CACHE) ---
@st.cache_data(ttl=3600)
def fetch_data_pro(ticker):
    try:
        # Descargamos los precios. multi_level_index=False evita errores de formato.
        hist = yf.download(ticker, period="5y", auto_adjust=True, multi_level_index=False)
        # Obtenemos los fundamentales por separado
        asset = yf.Ticker(ticker)
        info = asset.info
        return hist, info
    except Exception as e:
        return pd.DataFrame(), {}

# --- LÓGICA DE LA TERMINAL ---
st.title("🏛️ Gorostiaga Research - Terminal Quant")
ticker_input = st.sidebar.text_input("📍 Asset Ticker", "AAPL")

hist, info = fetch_data_pro(ticker_input)

if not hist.empty:
    last_price = float(hist['Close'].iloc[-1])
    
    # --- PANEL DE RATIOS ---
    st.subheader("📊 Fundamental & Market Metrics")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("P/E Forward", f"{info.get('forwardPE', 'N/A')}x")
    f2.metric("PEG Ratio", f"{info.get('pegRatio', 'N/A')}")
    f3.metric("P/B Ratio", f"{info.get('priceToBook', 'N/A')}x")
    f4.metric("Beta (1Y)", f"{info.get('beta', 'N/A')}")

    # --- ANÁLISIS CUANTITATIVO ---
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
    mu, sigma = np.mean(returns), np.std(returns)
    
    # Monte Carlo
    st.markdown("---")
    st.header("🎲 Proyección Estocástica (Monte Carlo)")
    days, sims = 252, 1000
    daily_yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
    paths = np.zeros_like(daily_yields); paths[0] = last_price
    for t in range(1, days): paths[t] = paths[t-1] * daily_yields[t]

    fig = go.Figure()
    for i in range(50):
        fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=0.5), opacity=0.1, showlegend=False))
    
    p75 = np.percentile(paths[-1], 75)
    fig.add_trace(go.Scatter(y=[p75]*days, name="P75 Target", line=dict(color='cyan', width=2)))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # --- VERDICTO ---
    prob_gain = (paths[-1] > last_price).mean() * 100
    sharpe = (mu * 252) / (sigma * np.sqrt(252))
    
    st.subheader("🏁 Research Recommendation")
    if prob_gain > 60 and sharpe > 0.5:
        st.success("**VERDICTO: COMPRAR / SOBREPONDERAR**")
    elif prob_gain < 45:
        st.error("**VERDICTO: VENDER / EVITAR**")
    else:
        st.warning("**VERDICTO: MANTENER (HOLD)**")
else:
    st.error("No se pudieron obtener datos. Verifique el ticker o intente más tarde.")
