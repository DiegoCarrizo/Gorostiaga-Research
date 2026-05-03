import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from fpdf import FPDF
import requests_cache

# --- SESIÓN ROBUSTA ---
session = requests_cache.CachedSession('yfinance.cache')
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
})

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

# --- CARGA DE DATOS SEPARADA ---
@st.cache_data(ttl=3600)
def fetch_history(ticker):
    # El historial rara vez se bloquea
    return yf.download(ticker, period="5y", auto_adjust=True, multi_level_index=False, session=session)

@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker):
    # Este es el punto crítico de bloqueo
    try:
        asset = yf.Ticker(ticker, session=session)
        return asset.info
    except:
        return {}

# --- LÓGICA PRINCIPAL ---
st.title("🏛️ Gorostiaga Research - Terminal Quant")
ticker_input = st.sidebar.text_input("📍 Asset Ticker", "AAPL")

hist = fetch_history(ticker_input)
info = fetch_fundamentals(ticker_input)

if not hist.empty:
    last_price = float(hist['Close'].iloc[-1])
    
    # Ratios (si fallan por rate limit, mostramos N/A pero la app sigue)
    st.subheader("📊 Fundamental Valuation")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("P/E Forward", f"{info.get('forwardPE', 'N/A')}x")
    f2.metric("PEG Ratio", f"{info.get('pegRatio', 'N/A')}")
    f3.metric("P/B Ratio", f"{info.get('priceToBook', 'N/A')}x")
    f4.metric("Beta", f"{info.get('beta', 'N/A')}")

    # --- MONTE CARLO ---
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
    mu, sigma = np.mean(returns), np.std(returns)
    
    st.markdown("---")
    st.header("🎲 Proyección Estocástica (Monte Carlo)")
    days, sims = 252, 1000
    daily_yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
    paths = np.zeros_like(daily_yields); paths[0] = last_price
    for t in range(1, days): paths[t] = paths[t-1] * daily_yields[t]

    fig = go.Figure()
    for i in range(50): # Dibujamos menos rutas para velocidad
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
    st.error("Esperando datos de Yahoo Finance. Reintente en unos minutos.")
