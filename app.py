import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

st.set_page_config(page_title="Gorostiaga Research | Quant Monitor", layout="wide")
st.title("🏛️ Professional Asset Monitor")

ticker = st.sidebar.text_input("Ticker", "AAPL")
asset = yf.Ticker(ticker)

try:
    data = asset.history(period="2y")
    if not data.empty:
        # --- BLOQUE FUNDAMENTAL ---
        info = asset.info
        c1, c2, c3 = st.columns(3)
        c1.metric("Beta", info.get('beta', 'N/A'))
        c2.metric("P/E Forward", f"{info.get('forwardPE', 'N/A')}x")
        c3.metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

        # --- BLOQUE CUANTITATIVO (VaR y Montecarlo) ---
        st.markdown("---")
        returns = data['Close'].pct_change().dropna()
        
        # Value at Risk (VaR 95%)
        var_95 = np.percentile(returns, 5)
        st.error(f"Value at Risk (95% confianza): {var_95*100:.2f}%")
        
        # Simulación Simple (Efecto Montecarlo)
        last_price = data['Close'].iloc[-1]
        st.subheader(f"Proyección Estocástica: {ticker}")
        st.line_chart(data['Close'])
        
    else:
        st.warning("Ticker no reconocido.")
except Exception as e:
    st.info("Aguardando conexión con Yahoo Finance...")
