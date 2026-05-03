import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- CONFIGURACIÓN ESTÉTICA ---
st.set_page_config(page_title="Gorostiaga Research | Stock Screener Pro", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 24px; }
    div[data-testid="stMetric"] { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stDataFrame { background-color: #161b22; border-radius: 10px; }
    h1, h2, h3, p, span { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DE MOAT Y RATING ---
def calcular_moat(info):
    # Basado en márgenes operativos y retornos sobre capital (ROE/ROIC)
    om = info.get('operatingMargins', 0)
    roe = info.get('returnOnEquity', 0)
    if om > 0.20 and roe > 0.15: return "Fuerte"
    if om > 0.10: return "Moderado"
    return "Débil"

def calcular_riesgo_rating(info, var_val):
    # Rating 1-10 basado en Deuda/Equity y Volatilidad (VaR)
    debt_equity = info.get('debtToEquity', 100) / 100
    riesgo = 5 + (debt_equity * 2) + (abs(var_val) * 10)
    return min(10, max(1, round(riesgo)))

# --- MOTOR DE ANÁLISIS ---
@st.cache_data(ttl=3600)
def analizar_seleccion(tickers):
    resultados = []
    for t in tickers:
        try:
            asset = yf.Ticker(t)
            info = asset.info
            hist = asset.history(period="5y")
            if hist.empty: continue
            
            # Análisis Cuantitativo
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values
            mu, sigma = np.mean(returns), np.std(returns)
            last_price = hist['Close'].iloc[-1]
            
            # Montecarlo P75
            days = 252
            sims = 500
            yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
            paths = np.zeros_like(yields); paths[0] = last_price
            for i in range(1, days): paths[i] = paths[i-1] * yields[i]
            p75 = np.percentile(paths[-1], 75)
            
            # VaR
            var_95 = np.percentile(returns, 5)
            
            resultados.append({
                "Ticker": t,
                "Precio": round(last_price, 2),
                "P/E Forward": info.get('forwardPE', 0),
                "Moat": calcular_moat(info),
                "Deuda/Capital": f"{info.get('debtToEquity', 0)}%",
                "Div. Yield": f"{info.get('dividendYield', 0)*100:.2f}%",
                "Target P75": round(p75, 2),
                "VaR (%)": round(var_95 * 100, 2),
                "Rating Riesgo": calcular_riesgo_rating(info, var_95),
                "Entrada": round(last_price * 0.95, 2),
                "Stop Loss": round(last_price * (1 + var_95 * 1.5), 2)
            })
        except: continue
    return pd.DataFrame(resultados)

# --- INTERFAZ ---
st.title("🏛️ Stock Screener TOP 10 - Gorostiaga Research")
st.markdown("Análisis multifactorial: Valuación, Solvencia, Moat y Proyección Estocástica.")

universo = st.sidebar.text_area("Lista de Tickers (separados por coma)", "AAPL, MSFT, GOOGL, META, TSLA, NVDA, BRK-B, V, JNJ, WMT")
tickers_list = [x.strip().upper() for x in universo.split(",")]

if st.button("🚀 Ejecutar Análisis de Selección"):
    with st.spinner("Procesando modelos estocásticos y fundamentales..."):
        df_final = analizar_seleccion(tickers_list)
        
        if not df_final.empty:
            # Ordenar por el mejor Rating de Riesgo y P/E
            df_final = df_final.sort_values(by=["Rating Riesgo", "P/E Forward"], ascending=[True, True]).head(10)
            
            st.header("🏆 TOP 10 Selección Research")
            st.table(df_final.style.format({"P/E Forward": "{:.2f}", "VaR (%)": "{:.2f}%"}))
            
            # Visualización Comparativa
            st.markdown("---")
            st.subheader("📊 Relación Riesgo / Retorno Esperado (P75)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_final["VaR (%)"], y=df_final["Target P75"],
                mode='markers+text', text=df_final["Ticker"],
                marker=dict(size=15, color=df_final["Rating Riesgo"], colorscale='Viridis', showscale=True),
                textposition="top center"
            ))
            fig.update_layout(title="Abanico de Oportunidades (Eje X: Riesgo VaR | Eje Y: Objetivo P75)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # Razonamiento de Rating
            with st.expander("ℹ️ Metodología del Rating de Riesgo"):
                st.write("El Rating (1-10) se calcula combinando el apalancamiento financiero (Deuda/Capital) con la volatilidad extrema capturada por el VaR histórico. Un rating de 1 indica activos con balances 'Misesianos' (sólidos) y baja volatilidad.")
        else:
            st.error("No se pudieron recuperar datos. Verifique los tickers.")
