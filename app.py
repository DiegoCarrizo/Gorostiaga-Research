import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- CONFIGURACIÓN VISUAL GOROSTIAGA ---
st.set_page_config(page_title="Gorostiaga Research | Terminal Multifuente", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 28px; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 14px; }
    div[data-testid="stMetric"] { 
        background-color: #161b22; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #30363d;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DE DATOS RESILIENTE (BACKUP ESTRATÉGICO) ---
@st.cache_data(ttl=3600)
def obtener_datos_resilientes(ticker):
    # Intentamos fuente principal (Yahoo Finance)
    try:
        data = yf.download(ticker, period="5y", auto_adjust=True, multi_level_index=False)
        asset = yf.Ticker(ticker)
        if data.empty:
            raise ValueError("Datos vacíos")
        return data, asset.info, "Yahoo Finance (Online)"
    except:
        # Aquí es donde el monitor sobrevive: 
        # Si Yahoo falla, devolvemos estructuras vacías para que la app no explote
        return pd.DataFrame(), {}, "⚠️ Error de Conexión (Modo Offline)"

# --- PANEL DE CONTROL ---
st.title("🏛️ Gorostiaga Research - Monitor Quant Multifuente")
ticker_input = st.sidebar.text_input("📍 Ticker del Activo", "AAPL").upper()

# Generación de Links Externos (Fuentes Secundarias de Consulta Directa)
st.sidebar.subheader("🔗 Fuentes de Respaldo")
st.sidebar.info("Si el monitor falla, use estos enlaces para validar datos en tiempo real:")
st.sidebar.markdown(f"[📈 Google Finance](https://www.google.com/finance/quote/{ticker_input}:NASDAQ)")
st.sidebar.markdown(f"[📊 TradingView](https://es.tradingview.com/symbols/{ticker_input}/)")
st.sidebar.markdown(f"[💼 Bloomberg](https://www.bloomberg.com/quote/{ticker_input}:US)")

hist, info, fuente_status = obtener_datos_resilientes(ticker_input)
st.sidebar.write(f"**Estado:** {fuente_status}")

if not hist.empty:
    precio_actual = float(hist['Close'].iloc[-1])
    
    # --- RATIOS FUNDAMENTALES ---
    st.subheader("📊 Ratios de Valuación (Métricas Bloomberg)")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("P/E Forward", f"{info.get('forwardPE', 'N/A')}x")
    f2.metric("PEG Ratio", f"{info.get('pegRatio', 'N/A')}")
    f3.metric("P/B Ratio", f"{info.get('priceToBook', 'N/A')}x")
    f4.metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

    # --- ANÁLISIS DE RIESGO ---
    st.markdown("---")
    st.header("📈 Riesgo y Solvencia")
    r1, r2, r3 = st.columns(3)
    
    retornos = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
    mu, sigma = np.mean(retornos), np.std(retornos)
    
    r1.metric("Beta (Market)", info.get('beta', 'N/A'))
    var_95 = np.percentile(retornos, 5)
    r2.metric("VaR Diario (95%)", f"{var_95*100:.2f}%")
    r3.metric("Sharpe Ratio", f"{(mu*252)/(sigma*np.sqrt(252)):.2f}")

    # --- MONTE CARLO ---
    st.header("🎲 Proyección Estocástica (Monte Carlo)")
    dias, sims = 252, 500
    daily_yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((dias, sims)))
    paths = np.zeros_like(daily_yields); paths[0] = precio_actual
    for t in range(1, dias): paths[t] = paths[t-1] * daily_yields[t]

    fig = go.Figure()
    for i in range(50):
        fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=0.5), opacity=0.1, showlegend=False))
    
    p75 = np.percentile(paths[-1], 75)
    fig.add_trace(go.Scatter(y=[p75]*dias, name="Target P75", line=dict(color='cyan', width=2)))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # --- VERDICTO ---
    prob_gain = (paths[-1] > precio_actual).mean() * 100
    st.markdown("---")
    st.subheader("🏁 Veredicto Final Gorostiaga Research")
    if prob_gain > 60:
        st.success(f"**RECOMENDACIÓN: COMPRAR / SOBREPONDERAR** (Probabilidad de Ganancia: {prob_gain:.1f}%)")
    else:
        st.warning(f"**RECOMENDACIÓN: NEUTRAL / MANTENER** (Probabilidad de Ganancia: {prob_gain:.1f}%)")

else:
    # SI FALLA YFINANCE, ACTIVAMOS EL MODO DE EMERGENCIA
    st.error("🚨 Yahoo Finance está bloqueado o fuera de línea.")
    st.warning("Usa los enlaces de la izquierda para obtener el precio actual y los ratios manuales.")
    
    # Permitimos entrada manual para no detener el análisis
    precio_manual = st.number_input("Introduzca Precio Actual (Manual) para ejecutar Monte Carlo", value=0.0)
    if precio_manual > 0:
        st.info("Ejecutando simulación basada en parámetros históricos genéricos...")
        # Lógica de respaldo con parámetros estándar de mercado si no hay data
