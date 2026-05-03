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
    om = info.get('operatingMargins', 0)
    roe = info.get('returnOnEquity', 0)
    if om > 0.20 and roe > 0.15: return "Fuerte"
    if om > 0.10: return "Moderado"
    return "Débil"

def calcular_riesgo_rating(info, var_val):
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
            days, sims = 252, 500
            yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
            paths = np.zeros_like(yields); paths[0] = last_price
            for i in range(1, days): paths[i] = paths[i-1] * yields[i]
            p75 = np.percentile(paths[-1], 75)
            
            # VaR y Métricas de Valuación
            var_95 = np.percentile(returns, 5)
            
            # Ratios de Valuación (Nuevos elementos)
            pe_fwd = info.get('forwardPE', 0)
            sector_pe = info.get('trailingPE', pe_fwd) # Fallback al actual
            
            resultados.append({
                "Ticker": t,
                "Precio": round(last_price, 2),
                "P/E Fwd": pe_fwd,
                "PEG Ratio": info.get('pegRatio', 0),
                "P/B Ratio": info.get('priceToBook', 0),
                "P/S Ratio": info.get('priceToSalesTrailing12Months', 0),
                "Moat": calcular_moat(info),
                "Sector": info.get('sector', 'N/A'),
                "Deuda/Cap": f"{info.get('debtToEquity', 0)}%",
                "Div. Yield": f"{info.get('dividendYield', 0)*100:.2f}%",
                "Target P75": round(p75, 2),
                "VaR (%)": round(var_95 * 100, 2),
                "Rating Riesgo": calcular_riesgo_rating(info, var_95),
                "Entrada": round(last_price * 0.95, 2),
                "Stop Loss": round(last_price * (1 + var_95 * 1.5), 2)
            })
        except: continue
    return pd.DataFrame(resultados)
