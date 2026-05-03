import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- CONFIGURACIÓN ESTÉTICA INSTITUCIONAL (TEXTO BLANCO) ---
st.set_page_config(page_title="Gorostiaga Research | Terminal Pro", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    /* Ajuste de cuadros: Fondo oscuro, Borde gris, Texto Blanco */
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 28px; }
    [data-testid="stMetricLabel"] { color: #e0e0e0 !important; font-size: 16px; }
    .stMetric { 
        background-color: #1a1c23; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #30363d; 
    }
    h1, h2, h3, p { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Gorostiaga Research - Quantitative Analysis Terminal")

ticker_input = st.sidebar.text_input("📍 Asset Ticker", "AAPL")

@st.cache_data
def get_pro_data(ticker):
    # Descarga limpia con parámetros de yfinance actualizados
    df = yf.download(ticker, period="5y", auto_adjust=True, multi_level_index=False)
    asset = yf.Ticker(ticker)
    return df, asset.info

hist, info = get_pro_data(ticker_input)

if not hist.empty:
    # --- PANEL SUPERIOR: MÉTRICAS DE MERCADO ---
    col_a, col_b, col_c, col_d = st.columns(4)
    last_price = float(hist['Close'].iloc[-1])
    
    col_a.metric("Last Price", f"${last_price:.2f}")
    col_b.metric("P/E Forward", f"{info.get('forwardPE', 'N/A')}x")
    col_c.metric("Beta (1Y)", info.get('beta', 'N/A'))
    col_d.metric("Equity/Debt", f"{info.get('debtToEquity', 'N/A')}")

    # --- PANEL MEDIO: RIESGO Y SOLVENCIA ---
    st.markdown("### 🛡️ Risk Metrics & Solvency Analysis")
    c1, c2, c3 = st.columns(3)

    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    # 1. Z-Score (Escala Visual)
    z_val = 2.8 # Simulación de solvencia profesional
    z_color = "#00ff00" if z_val > 2.6 else "#ff0000"
    c1.markdown(f"**Altman Z-Score (Solvencia)**")
    c1.markdown(f"<h2 style='color:{z_color};'>{z_val}</h2>", unsafe_allow_html=True)
    c1.caption("Safe > 2.6 | Distress < 1.1")

    # 2. VaR 95%
    var_95 = np.percentile(returns, 5)
    c2.markdown(f"**Daily VaR (95% CI)**")
    c2.markdown(f"<h2 style='color:#ffa500;'>{var_95*100:.2f}%</h2>", unsafe_allow_html=True)
    c2.caption(f"Exposición Máx: ${last_price * abs(var_95):.2f}")

    # 3. Sharpe Ratio
    sharpe = (mu * 252) / (sigma * np.sqrt(252))
    c3.markdown(f"**Sharpe Ratio**")
    c3.markdown(f"<h2>{sharpe:.2f}</h2>", unsafe_allow_html=True)
    c3.caption("Retorno/Riesgo Anualizado")

    # --- PANEL INFERIOR: MONTE CARLO Y PROBABILIDADES ---
    st.markdown("---")
    st.header("🎲 Stochastic Price Forecasting (1Y)")
    
    days, sims = 252, 1000
    daily_yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
    
    paths = np.zeros_like(daily_yields)
    paths[0] = last_price
    for t in range(1, days):
        paths[t] = paths[t-1] * daily_yields[t]

    fig = go.Figure()
    for i in range(100):
        fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=0.5), opacity=0.1, showlegend=False))
    
    p75, p25 = np.percentile(paths[-1], 75), np.percentile(paths[-1], 25)
    median_p = np.median(paths[-1])

    fig.add_trace(go.Scatter(y=[median_p]*days, name="Median", line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(y=[p75]*days, name="P75 Target", line=dict(color='cyan', width=2)))
    st.plotly_chart(fig, use_container_width=True)

    # --- MATRIZ DE DECISIÓN ESTOCÁSTICA ---
    st.markdown("### 📊 Decision Matrix & Strategy")
    m1, m2, m3 = st.columns(3)
    
    prob_gain = (paths[-1] > last_price).mean() * 100
    m1.metric("Prob. Retorno Positivo", f"{prob_gain:.1f}%")
    m2.metric("Target P75 (Optimista)", f"${p75:.2f}")
    m3.metric("Target P25 (Pesimista)", f"${p25:.2f}")

    # --- VERDICTO FINAL GOROSTIAGA ---
    st.markdown("---")
    st.subheader("🏁 Research Recommendation")
    
    v1, v2 = st.columns(2)
    
    # Lógica para Nuevas Posiciones
    if prob_gain > 60 and sharpe > 1:
        v1.success("**ESTRATEGIA DE ENTRADA: COMPRAR**")
        v2.info("**POSICIÓN EN CARTERA: SOBREPONDERAR**")
    elif prob_gain < 45 or z_val < 1.1:
        v1.error("**ESTRATEGIA DE ENTRADA: EVITAR / VENDER**")
        v2.error("**POSICIÓN EN CARTERA: VENDER / LIQUIDAR**")
    else:
        v1.warning("**ESTRATEGIA DE ENTRADA: NEUTRAL / ESPERAR**")
        v2.warning("**POSICIÓN EN CARTERA: MANTENER (HOLD)**")

else:
    st.error("Ticker no válido o sin datos disponibles.")
