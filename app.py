import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Gorostiaga Research | Asset Monitor", layout="wide")
st.title("🏛️ Professional Asset Analytics Monitor")

ticker_input = st.sidebar.text_input("Ticker (ej: AAPL, AL30.BA, GGAL)", "AAPL")
period = st.sidebar.selectbox("Horizonte de Análisis", ["1y", "2y", "5y", "10y"])

@st.cache_data
def get_historical_data(ticker, p):
    return yf.download(ticker, period=p)

# Manejo de datos
asset = yf.Ticker(ticker_input)
hist = get_historical_data(ticker_input, "5y")

if not hist.empty:
    # --- ANÁLISIS FUNDAMENTAL ---
    st.header("📊 Análisis Fundamental")
    col1, col2, col3 = st.columns(3)
    
    info = asset.info
    # Usamos .get() para evitar que la app explote si no hay datos
    div_yield = info.get('dividendYield', 0) * 100
    p_e_forward = info.get('forwardPE', 0)
    beta = info.get('beta', 0)
    
    col1.metric("Dividend Yield", f"{div_yield:.2f}%")
    col2.metric("P/E Forward", f"{p_e_forward:.2f}x")
    col3.metric("Beta", f"{beta:.2f}")

    # --- MÉTRICAS DE RIESGO ---
    st.markdown("---")
    st.header("📈 Riesgo y Solvencia")
    
    # Cálculo de rendimientos logarítmicos para mayor precisión doctoral
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    var_95 = np.percentile(returns, 5)
    
    st.error(f"**Value at Risk (95% Confianza):** {var_95*100:.2f}% (Pérdida máxima diaria esperada)")

    # --- MONTECARLO ---
    st.header("🎲 Simulación de Montecarlo (1 Año)")
    days = 252
    sims = 100
    
    mu = returns.mean()
    sigma = returns.std()
    last_p = hist['Close'].iloc[-1]
    
    # Simulación vectorizada (más rápida)
    dt = 1
    stoch_vol = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal((days, sims)))
    results = np.zeros_like(stoch_vol)
    results[0] = last_p
    for t in range(1, days):
        results[t] = results[t-1] * stoch_vol[t]

    fig = go.Figure()
    for i in range(sims):
        fig.add_trace(go.Scatter(y=results[:, i], mode='lines', line=dict(width=1), opacity=0.2, showlegend=False))
    
    st.plotly_chart(fig, use_container_width=True)

    # --- VERDICTO ---
    st.markdown("---")
    score = 0
    if var_95 > -0.03: score += 1
    if p_e_forward > 0 and p_e_forward < 25: score += 1
    if div_yield > 2: score += 1

    if score >= 2:
        st.success("📝 VERDICTO: COMPRAR / MANTENER")
    else:
        st.warning("⚠️ VERDICTO: PRECAUCIÓN / EVITAR")
else:
    st.error("No se encontraron datos para el Ticker ingresado.")
