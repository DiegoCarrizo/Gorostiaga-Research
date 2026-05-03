import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- CONFIGURACIÓN VISUAL GOROSTIAGA ---
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

@st.cache_data(ttl=3600)
def fetch_data_safe(ticker):
    try:
        # Descargamos solo precios (esto casi nunca se bloquea)
        hist = yf.download(ticker, period="5y", auto_adjust=True, multi_level_index=False)
        # Intentamos traer info, pero con manejo de error silencioso
        try:
            asset = yf.Ticker(ticker)
            info = asset.info
        except:
            info = {}
        return hist, info
    except:
        return pd.DataFrame(), {}

# --- INTERFAZ ---
st.title("🏛️ Gorostiaga Research - Terminal Quant")
ticker_input = st.sidebar.text_input("📍 Asset Ticker", "AAPL").upper()

hist, info = fetch_data_safe(ticker_input)

if not hist.empty:
    last_price = float(hist['Close'].iloc[-1])
    
    # --- PANEL DE RATIOS (CON FALLBACK) ---
    st.subheader("📊 Fundamental Metrics")
    f1, f2, f3, f4 = st.columns(4)
    
    # Si info falla, mostramos N/A pero la app sigue
    f1.metric("P/E Forward", f"{info.get('forwardPE', 'N/A')}x")
    f2.metric("PEG Ratio", f"{info.get('pegRatio', 'N/A')}")
    f3.metric("P/B Ratio", f"{info.get('priceToBook', 'N/A')}x")
    
    # Calculamos Beta manualmente si no viene en info
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    beta_val = info.get('beta', 'N/A')
    f4.metric("Beta (Market)", beta_val)

    # --- ANÁLISIS CUANTITATIVO ---
    st.markdown("---")
    st.header("🎲 Proyección Estocástica (Monte Carlo)")
    
    returns_val = returns.values
    mu, sigma = np.mean(returns_val), np.std(returns_val)
    days, sims = 252, 1000
    
    daily_yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
    paths = np.zeros_like(daily_yields); paths[0] = last_price
    for t in range(1, days): paths[t] = paths[t-1] * daily_yields[t]

    fig = go.Figure()
    for i in range(50):
        fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=0.5), opacity=0.1, showlegend=False))
    
    p75 = np.percentile(paths[-1], 75)
    fig.add_trace(go.Scatter(y=[p75]*days, name="Target P75", line=dict(color='cyan', width=2)))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # --- VERDICTO FINAL ---
    prob_gain = (paths[-1] > last_price).mean() * 100
    sharpe = (mu * 252) / (sigma * np.sqrt(252))
    
    st.subheader("🏁 Research Recommendation")
    c1, c2 = st.columns(2)
    if prob_gain > 60 and sharpe > 0.5:
        c1.success("**ESTRATEGIA: COMPRAR / SOBREPONDERAR**")
    elif prob_gain < 45:
        c1.error("**ESTRATEGIA: VENDER / EVITAR**")
    else:
        c1.warning("**ESTRATEGIA: MANTENER (HOLD)**")
    
    c2.metric("Prob. Retorno Positivo", f"{prob_gain:.1f}%")

else:
    st.error("⚠️ Error de Conexión: Yahoo Finance ha limitado las peticiones. Por favor, intente con otro ticker o aguarde 5 minutos.")
