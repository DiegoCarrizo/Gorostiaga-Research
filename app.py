import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Gorostiaga Research | Quant Analytics", layout="wide")
st.title("🏛️ Professional Asset Analytics Monitor")

ticker_input = st.sidebar.text_input("Ticker (ej: AAPL, AL30.BA, GGAL)", "AAPL")

@st.cache_data
def get_historical_data(ticker):
    # Descargamos los datos y nos aseguramos de que sean numéricos
    df = yf.download(ticker, period="5y")
    return df

asset = yf.Ticker(ticker_input)
hist = get_historical_data(ticker_input)

if not hist.empty:
    # --- MÉTRICAS FUNDAMENTALES ---
    info = asset.info
    col1, col2, col3 = st.columns(3)
    col1.metric("P/E Forward", f"{info.get('forwardPE', 'N/A')}x")
    col2.metric("Beta", info.get('beta', 'N/A'))
    col3.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

    # --- ANÁLISIS CUANTITATIVO ---
    # Convertimos a valores numéricos para evitar el ValueError de longitud
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
    
    mu = np.mean(returns)
    sigma = np.std(returns)
    last_price = float(hist['Close'].iloc[-1])

    # --- SIMULACIÓN DE MONTECARLO (Ajustada) ---
    st.markdown("---")
    st.header("🎲 Proyección de Montecarlo (1 año)")
    
    days = 252
    sims = 100
    
    # Generamos los rendimientos simulados usando solo escalares numéricos
    daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
    
    # Construimos las rutas de precios
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = last_price
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]

    # Gráfico Profesional con Plotly
    fig = go.Figure()
    for i in range(sims):
        fig.add_trace(go.Scatter(y=price_paths[:, i], mode='lines', 
                                 line=dict(width=1), opacity=0.2, showlegend=False))
    
    st.plotly_chart(fig, use_container_width=True)

    # --- MÉTRICA DE RIESGO VaR ---
    var_95 = np.percentile(returns, 5)
    st.error(f"**Value at Risk (95% confianza):** {var_95*100:.2f}% (Pérdida máxima diaria esperada)")

else:
    st.error("No se encontraron datos. Verifique el Ticker en Yahoo Finance.")
