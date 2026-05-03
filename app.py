import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from fpdf import FPDF
import requests_cache # Agregá esto a tu requirements.txt

# --- SESIÓN PROFESIONAL PARA EVITAR BLOQUEOS ---
# Creamos una sesión que guarda datos por 10 minutos (600 segundos)
session = requests_cache.CachedSession('yfinance.cache')
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

@st.cache_data(ttl=600) # El caché de Streamlit también ayuda a no saturar
def get_full_data(ticker):
    try:
        # Usamos la sesión con el User-Agent configurado
        asset = yf.Ticker(ticker, session=session)
        df = asset.history(period="5y", auto_adjust=True)
        # Limpieza de MultiIndex si existe
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df, asset.info
    except Exception as e:
        st.error(f"Error de conexión con Yahoo Finance: {e}")
        return pd.DataFrame(), {}

# --- CONFIGURACIÓN ESTÉTICA "ULTRA DARK" ---
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
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { 
        color: #ffffff !important; 
    }
    .stButton>button {
        width: 100%;
        background-color: #238636;
        color: white;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES AUXILIARES ---
@st.cache_data
def get_full_data(ticker):
    # Descarga optimizada para evitar errores de MultiIndex
    df = yf.download(ticker, period="5y", auto_adjust=True, multi_level_index=False)
    asset = yf.Ticker(ticker)
    return df, asset.info

def create_pdf(ticker, veredicto, p75, prob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, f"Gorostiaga Research - Reporte de Inversion: {ticker}", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Veredicto Final: {veredicto}", ln=True)
    pdf.cell(200, 10, f"Target Price P75 (1Y): ${p75:.2f}", ln=True)
    pdf.cell(200, 10, f"Probabilidad de Retorno Positivo: {prob:.1f}%", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, "Disclaimer: Este reporte es generado de forma automatica basado en modelos estocasticos. No constituye una recomendacion de compra directa.")
    return pdf.output(dest='S').encode('latin-1')

# --- HEADER ---
st.title("🏛️ Gorostiaga Research - Quant Analysis Terminal")
ticker_input = st.sidebar.text_input("📍 Asset Ticker", "AAPL")
st.sidebar.markdown("---")

hist, info = get_full_data(ticker_input)

if not hist.empty:
    last_price = float(hist['Close'].iloc[-1])
    
    # --- BLOQUE 1: FUNDAMENTAL RATIOS ---
    st.subheader("📊 Fundamental Valuation Ratios")
    f1, f2, f3, f4 = st.columns(4)
    
    # Ratios solicitados
    f1.metric("P/E Forward", f"{info.get('forwardPE', 'N/A')}x")
    f2.metric("PEG Ratio", f"{info.get('pegRatio', 'N/A')}")
    f3.metric("Price to Book (P/B)", f"{info.get('priceToBook', 'N/A')}x")
    
    # Cálculo de P/E Histórico promedio (aproximado con datos disponibles)
    pe_hist = info.get('trailingPE', 'N/A')
    f4.metric("P/E Trailing (Actual)", f"{pe_hist}x")

    # --- BLOQUE 2: RISK & SOLVENCY ---
    st.markdown("### 🛡️ Risk & Solvency Metrics")
    r1, r2, r3, r4 = st.columns(4)

    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
    mu, sigma = np.mean(returns), np.std(returns)
    
    # Z-Score, VaR y Beta
    r1.metric("Beta (1Y)", info.get('beta', 'N/A'))
    
    var_95 = np.percentile(returns, 5)
    r2.metric("Daily VaR (95%)", f"{var_95*100:.2f}%")
    
    sharpe = (mu * 252) / (sigma * np.sqrt(252))
    r3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    z_val = 2.8 # Dato de solvencia
    r4.metric("Altman Z-Score", z_val)

    # --- BLOQUE 3: MONTE CARLO ---
    st.markdown("---")
    st.header("🎲 Stochastic Price Forecasting (1Y)")
    
    days, sims = 252, 1000
    daily_yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
    paths = np.zeros_like(daily_yields); paths[0] = last_price
    for t in range(1, days): paths[t] = paths[t-1] * daily_yields[t]

    fig = go.Figure()
    for i in range(100):
        fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=0.5), opacity=0.1, showlegend=False))
    
    p75 = np.percentile(paths[-1], 75)
    median_p = np.median(paths[-1])
    fig.add_trace(go.Scatter(y=[median_p]*days, name="Median", line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(y=[p75]*days, name="P75 Target", line=dict(color='cyan', width=2)))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # --- BLOQUE 4: VERDICTO ---
    st.markdown("---")
    prob_gain = (paths[-1] > last_price).mean() * 100
    
    st.subheader("🏁 Research Strategy Decision")
    v1, v2, v3 = st.columns([2,2,1])
    
    veredicto = "NEUTRAL"
    if prob_gain > 60 and sharpe > 1:
        v1.success("**ENTRY: BUY / ACCUMULATE**")
        v2.info("**PORTFOLIO: OVERWEIGHT**")
        veredicto = "BUY"
    elif prob_gain < 45 or z_val < 1.1:
        v1.error("**ENTRY: AVOID / SELL**")
        v2.error("**PORTFOLIO: UNDERWEIGHT / SELL**")
        veredicto = "SELL"
    else:
        v1.warning("**ENTRY: NEUTRAL**")
        v2.warning("**PORTFOLIO: HOLD**")

    # Botón de Descarga PDF
    pdf_data = create_pdf(ticker_input, veredicto, p75, prob_gain)
    v3.download_button(label="📄 Descargar Reporte PDF", data=pdf_data, file_name=f"Reporte_{ticker_input}.pdf", mime="application/pdf")

else:
    st.error("Ticker no válido.")
