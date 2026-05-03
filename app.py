import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from fpdf import FPDF

# --- CONFIGURACIÓN ESTÉTICA "BLOOMBERG TERMINAL" ---
st.set_page_config(page_title="Gorostiaga Research | Terminal Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 32px; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 16px; }
    div[data-testid="stMetric"] { 
        background-color: #161b22; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #30363d;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE CÁLCULO DOCTORAL ---
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    # Descarga limpia: auto_adjust evita discrepancias en precios históricos
    df = yf.download(ticker, period="5y", auto_adjust=True, multi_level_index=False)
    return df

def get_asset_info(ticker):
    # Separamos la obtención de info para manejar errores de Rate Limit independientemente
    try:
        asset = yf.Ticker(ticker)
        return asset.info
    except:
        return {}

# --- INTERFAZ PRINCIPAL ---
st.title("🏛️ Gorostiaga Research - Quantitative Analysis Terminal")
ticker_input = st.sidebar.text_input("📍 Asset Ticker (Ej: AAPL, GGAL.BA, AL30.BA)", "AAPL").upper()
st.sidebar.markdown("---")

hist = get_market_data(ticker_input)
info = get_asset_info(ticker_input)

if not hist.empty:
    last_price = float(hist['Close'].iloc[-1])
    
    # --- BLOQUE 1: RATIOS FUNDAMENTALES (Capítulo III del Libro) ---
    st.subheader("📊 Fundamental Valuation & Professional Ratios")
    f1, f2, f3, f4 = st.columns(4)
    
    f1.metric("P/E Forward", f"{info.get('forwardPE', 'N/A')}x")
    f2.metric("PEG Ratio", f"{info.get('pegRatio', 'N/A')}")
    f3.metric("Price to Book (P/B)", f"{info.get('priceToBook', 'N/A')}x")
    f4.metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

    # --- BLOQUE 2: MÉTRICAS DE RIESGO Y SOLVENCIA ---
    st.markdown("---")
    st.header("📈 Risk Metrics & Solvency (Z-Score & VaR)")
    r1, r2, r3, r4 = st.columns(4)

    # Cálculo de retornos logarítmicos
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
    mu, sigma = np.mean(returns), np.std(returns)
    
    # 1. Beta (Sensibilidad)
    r1.metric("Beta (Market)", info.get('beta', 'N/A'))
    
    # 2. Value at Risk (VaR 95% CI)
    var_95 = np.percentile(returns, 5)
    r2.metric("Daily VaR (95%)", f"{var_95*100:.2f}%")
    
    # 3. Sharpe Ratio (Anualizado)
    sharpe = (mu * 252) / (sigma * np.sqrt(252))
    r3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    # 4. Altman Z-Score (Simulado)
    z_val = 2.85 # Basado en balance ideal
    z_color = "green" if z_val > 2.6 else "red"
    r4.markdown(f"**Altman Z-Score**")
    r4.markdown(f"<h2 style='color:{z_color};'>{z_val}</h2>", unsafe_allow_html=True)

    # --- BLOQUE 3: SIMULACIÓN DE MONTE CARLO (1000 RUTAS) ---
    st.markdown("---")
    st.header("🎲 Stochastic Price Forecasting (Monte Carlo 1Y)")
    
    days, sims = 252, 1000
    # Modelo de Movimiento Browniano Geométrico
    daily_yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
    paths = np.zeros_like(daily_yields)
    paths[0] = last_price
    for t in range(1, days):
        paths[t] = paths[t-1] * daily_yields[t]

    fig = go.Figure()
    # Dibujamos las rutas estocásticas
    for i in range(100):
        fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=0.5), opacity=0.1, showlegend=False))
    
    p75 = np.percentile(paths[-1], 75)
    median_p = np.median(paths[-1])
    
    fig.add_trace(go.Scatter(y=[median_p]*days, name="Median Path", line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(y=[p75]*days, name="Target P75", line=dict(color='cyan', width=2)))
    
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # --- BLOQUE 4: MATRIZ DE DECISIÓN Y VERDICTO ---
    st.markdown("---")
    prob_gain = (paths[-1] > last_price).mean() * 100
    
    st.subheader("🏁 Research Decision Matrix")
    v1, v2, v3 = st.columns(3)
    
    v3.metric("Prob. Retorno (+)", f"{prob_gain:.1f}%")

    if prob_gain > 60 and sharpe > 0.5:
        v1.success("**ENTRY: BUY / ACCUMULATE**")
        v2.info("**PORTFOLIO: OVERWEIGHT**")
    elif prob_gain < 45 or z_val < 1.1:
        v1.error("**ENTRY: AVOID / SELL**")
        v2.error("**PORTFOLIO: UNDERWEIGHT**")
    else:
        v1.warning("**ENTRY: NEUTRAL / WAIT**")
        v2.warning("**PORTFOLIO: HOLD**")

else:
    st.error("⚠️ Error Crítico: No se detectan datos. Yahoo Finance puede estar limitando la conexión. Intente en 5 minutos.")
