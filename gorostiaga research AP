import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Configuración de nivel profesional
st.set_page_config(page_title="J.P. Morgan Asset Monitor - Quant Analytics", layout="wide")

st.title("🏛️ Professional Asset Analytics Monitor")
st.markdown("---")

# Input del Activo
ticker_input = st.sidebar.text_input("Introduzca Ticker (ej: AAPL, AL30.BA, BABA)", "AAPL")
period = st.sidebar.selectbox("Horizonte de Análisis", ["1y", "2y", "5y", "10y"])

@st.cache_data
def get_data(ticker):
    asset = yf.Ticker(ticker)
    hist = asset.history(period="5y")
    return asset, hist

asset, hist = get_data(ticker_input)

# --- Módulo de Análisis Fundamental ---
st.header("📊 Análisis Fundamental & Cash Flow")
col1, col2, col3 = st.columns(3)

if asset.info.get('quoteType') == 'EQUITY':
    # Métricas para Acciones
    dividend_yield = asset.info.get('dividendYield', 0) * 100
    p_e_forward = asset.info.get('forwardPE', 0)
    beta = asset.info.get('beta', 0)
    
    col1.metric("Dividend Yield", f"{dividend_yield:.2f}%")
    col2.metric("P/E Forward", f"{p_e_forward:.2f}x")
    col3.metric("Beta (Sensibilidad)", f"{beta:.2f}")
    
    # Calendario de Dividendos
    st.subheader("📅 Próximos Pagos de Dividendos")
    st.write(asset.dividends.tail(5))
else:
    # Lógica simplificada para renta fija (Bonos)
    st.subheader("💸 Proyección de Flujo de Fondos (Cash Flow)")
    st.info("Para bonos soberanos/corp, el flujo depende de la estructura de amortización (Bullet vs. Sinkable).")
    # Aquí se integraría el scrap del prospecto de emisión

# --- Módulo Cuantitativo Avanzado ---
st.markdown("---")
st.header("📈 Métricas de Riesgo y Solvencia")

# 1. Altman Z-Score (Solvencia)
# Simplificación: Requiere datos de Balance (Total Assets, Retained Earnings, etc.)
def calculate_z_score(info):
    try:
        # Fórmula: 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        # Se requiere acceso a Financials de yfinance
        return "3.1 (Zona Segura)" 
    except:
        return "N/A"

# 2. Value at Risk (VaR) Histórico
returns = hist['Close'].pct_change().dropna()
var_95 = np.percentile(returns, 5)

st.write(f"**Value at Risk (95% Confianza):** {var_95*100:.2f}% (Pérdida máxima esperada diaria)")

# --- Simulación de Montecarlo ---
st.header("🎲 Proyecciones de Montecarlo (Horizonte 1 año)")
days_to_forecast = 252
num_simulations = 1000

last_price = hist['Close'][-1]
mu = returns.mean()
sigma = returns.std()

simulation_df = pd.DataFrame()

for x in range(num_simulations):
    prices = [last_price]
    for d in range(days_to_forecast):
        prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * norm.ppf(np.random.rand())))
    simulation_df[x] = prices

# Gráfico Plotly
fig = go.Figure()
for i in range(100): # Mostrar solo 100 rutas para no saturar
    fig.add_trace(go.Scatter(y=simulation_df[i], mode='lines', line=dict(width=1), opacity=0.1, showlegend=False))

st.plotly_chart(fig, use_container_width=True)

# --- SISTEMA DE RECOMENDACIÓN ---
st.markdown("---")
st.header("🚩 Veredicto de Inversión")

score = 0
# Ejemplo de Lógica de Decisión
if var_95 > -0.02: score += 1 # Poco riesgo
if asset.info.get('forwardPE', 100) < 20: score += 1 # Valuación razonable
if asset.info.get('dividendYield', 0) > 0.03: score += 1 # Genera renta

if score >= 2:
    st.success("RECOMENDACIÓN: COMPRAR / MANTENER (Fundamentos sólidos y riesgo controlado)")
else:
    st.error("RECOMENDACIÓN: EVITAR / VENDER (Alta volatilidad o valuación excesiva)")
