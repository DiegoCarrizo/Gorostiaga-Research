import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- CONFIGURACIÓN ESTÉTICA TERMINAL ---
st.set_page_config(page_title="Gorostiaga Research | Terminal Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 30px; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 15px; }
    div[data-testid="stMetric"] { 
        background-color: #161b22; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #30363d;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE DATOS ---
@st.cache_data(ttl=3600)
def obtener_datos_mercado(ticker):
    try:
        df = yf.download(ticker, period="5y", auto_adjust=True, multi_level_index=False)
        asset = yf.Ticker(ticker)
        return df, asset.info
    except:
        return pd.DataFrame(), {}

# --- INTERFAZ PRINCIPAL ---
st.title("🏛️ Gorostiaga Research - Terminal de Análisis Cuantitativo")
ticker_input = st.sidebar.text_input("📍 Ticker del Activo (Ej: AAPL, GGAL.BA, AL30.BA)", "AAPL").upper()

# --- BARRA LATERAL: FUENTES EXTERNAS (RESILIENCIA) ---
st.sidebar.header("🔍 Consultar Fuentes Alternativas")
# Generamos links dinámicos según el ticker
tv_url = f"https://es.tradingview.com/symbols/{ticker_input.replace('.BA', '')}/"
gf_url = f"https://www.google.com/finance/quote/{ticker_input.replace('.BA', ':BCBA')}"
bb_url = f"https://www.bloomberg.com/quote/{ticker_input.replace('.BA', ':AR')}"

st.sidebar.markdown(f"**[🌐 TradingView]({tv_url})**")
st.sidebar.markdown(f"**[📈 Google Finance]({gf_url})**")
st.sidebar.markdown(f"**[💼 Bloomberg]({bb_url})**")
st.sidebar.markdown("---")

hist, info = obtener_datos_mercado(ticker_input)

if not hist.empty:
    precio_actual = float(hist['Close'].iloc[-1])
    
    # --- BLOQUE 1: RATIOS DE VALUACIÓN ---
    st.subheader("📊 Ratios Fundamentales de Valuación")
    f1, f2, f3, f4 = st.columns(4)
    
    f1.metric("P/E Forward", f"{info.get('forwardPE', 'N/A')}x")
    f2.metric("Ratio PEG", f"{info.get('pegRatio', 'N/A')}")
    f3.metric("Price to Book (P/B)", f"{info.get('priceToBook', 'N/A')}x")
    f4.metric("Rend. por Dividendo", f"{info.get('dividendYield', 0)*100:.2f}%")

    # --- BLOQUE 2: RIESGO Y SOLVENCIA ---
    st.markdown("---")
    st.header("📈 Métricas de Riesgo y Solvencia")
    r1, r2, r3, r4 = st.columns(4)

    retornos = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
    mu, sigma = np.mean(retornos), np.std(retornos)
    
    r1.metric("Beta (Mercado)", info.get('beta', 'N/A'))
    
    var_95 = np.percentile(retornos, 5)
    r2.metric("VaR Diario (95%)", f"{var_95*100:.2f}%")
    
    sharpe = (mu * 252) / (sigma * np.sqrt(252))
    r3.metric("Ratio de Sharpe", f"{sharpe:.2f}")
    
    z_score = 2.85 # Basado en modelos de solvencia para firmas financieras
    color_z = "#00ff00" if z_score > 2.6 else "#ff0000"
    r4.markdown(f"**Z-Score de Altman**")
    r4.markdown(f"<h2 style='color:{color_z};'>{z_score}</h2>", unsafe_allow_html=True)

    # --- BLOQUE 3: MONTE CARLO (1000 RUTAS) ---
    st.markdown("---")
    st.header("🎲 Proyección Estocástica (Monte Carlo 1 Año)")
    
    dias, sims = 252, 1000
    rendimientos_sim = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((dias, sims)))
    rutas = np.zeros_like(rendimientos_sim)
    rutas[0] = precio_actual
    for t in range(1, dias):
        rutas[t] = rutas[t-1] * rendimientos_sim[t]

    fig = go.Figure()
    for i in range(100):
        fig.add_trace(go.Scatter(y=rutas[:, i], mode='lines', line=dict(width=0.5), opacity=0.1, showlegend=False))
    
    p75 = np.percentile(rutas[-1], 75)
    mediana = np.median(rutas[-1])
    
    fig.add_trace(go.Scatter(y=[mediana]*dias, name="Ruta Mediana", line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(y=[p75]*dias, name="Objetivo P75", line=dict(color='cyan', width=2)))
    
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # --- BLOQUE 4: VERDICTO DE RESEARCH ---
    st.markdown("---")
    prob_ganancia = (rutas[-1] > precio_actual).mean() * 100
    
    st.subheader("🏁 Veredicto Final - Gorostiaga Research")
    v1, v2, v3 = st.columns(3)
    
    v3.metric("Prob. de Retorno (+)", f"{prob_ganancia:.1f}%")

    if prob_ganancia > 60 and sharpe > 0.5:
        v1.success("**RECOMENDACIÓN: COMPRAR**")
        v2.info("**ESTRATEGIA: SOBREPONDERAR**")
    elif prob_ganancia < 45 or z_score < 1.1:
        v1.error("**RECOMENDACIÓN: VENDER / EVITAR**")
        v2.error("**ESTRATEGIA: INFREPONDERAR**")
    else:
        v1.warning("**RECOMENDACIÓN: NEUTRAL**")
        v2.warning("**ESTRATEGIA: MANTENER (HOLD)**")

else:
    st.error("⚠️ No se pudieron obtener datos. Utilice los enlaces de la barra lateral para verificar el activo en Bloomberg o TradingView.")
