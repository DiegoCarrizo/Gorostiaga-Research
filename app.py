import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURACIÓN ESTÉTICA ---
st.set_page_config(page_title="Gorostiaga Research | Terminal Pro", layout="wide")

# Estilo CSS para asegurar visibilidad
st.markdown("""
    <style>
    .main { background-color: #0e1117 !important; color: #ffffff !important; }
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DE APOYO ---
def calcular_moat(info):
    try:
        om = info.get('operatingMargins', 0) or 0
        roe = info.get('returnOnEquity', 0) or 0
        if om > 0.20 and roe > 0.15: return "Fuerte"
        if om > 0.10: return "Moderado"
        return "Débil"
    except: return "N/A"

def calcular_riesgo_rating(info, var_val):
    try:
        debt_equity = (info.get('debtToEquity', 100) or 100) / 100
        riesgo = 5 + (debt_equity * 2) + (abs(var_val) * 10)
        return int(min(10, max(1, round(riesgo))))
    except: return 5

# --- MOTOR DE ANÁLISIS BLINDADO ---
@st.cache_data(ttl=3600)
def analizar_seleccion(tickers):
    resultados = []
    for t in tickers:
        try:
            asset = yf.Ticker(t)
            # Forzamos descarga limpia para evitar Multi-Index
            hist = yf.download(t, period="2y", progress=False, multi_level_index=False)
            if hist.empty: continue
            
            info = asset.info
            close = hist['Close']
            
            # Análisis Cuantitativo
            returns = np.log(close / close.shift(1)).dropna().values
            mu, sigma = np.mean(returns), np.std(returns)
            last_price = float(close.iloc[-1])
            
            # Montecarlo P75 (Simplificado para estabilidad)
            p75 = last_price * np.exp((mu - 0.5 * sigma**2) * 252 + sigma * np.sqrt(252) * 0.674)
            var_95 = np.percentile(returns, 5)
            
            # Ratios de Valuación
            pe_fwd = info.get('forwardPE', 0) or 0
            
            resultados.append({
                "Ticker": t,
                "Precio": round(last_price, 2),
                "P/E Fwd": float(pe_fwd),
                "PEG Ratio": float(info.get('pegRatio', 0) or 0),
                "P/B Ratio": float(info.get('priceToBook', 0) or 0),
                "Moat": calcular_moat(info),
                "Sector": info.get('sector', 'N/A'),
                "Deuda/Cap": f"{info.get('debtToEquity', 0) or 0}%",
                "Target P75": float(p75),
                "VaR (%)": float(var_95 * 100),
                "Rating Riesgo": calcular_riesgo_rating(info, var_95),
                "Entrada": round(last_price * 0.95, 2)
            })
        except Exception as e:
            st.sidebar.warning(f"Error analizando {t}: {e}")
            continue
    return pd.DataFrame(resultados)

# --- INTERFAZ ---
st.title("🏛️ Gorostiaga Research - Terminal Quant")
st.markdown("Análisis de Valuación y Riesgo Estocástico")

universo_input = st.sidebar.text_area("Lista de Tickers", "AAPL, MSFT, GOOGL, META, NVDA, GGAL, YPF, BMA, PAM")
tickers_list = [x.strip().upper() for x in universo_input.split(",") if x.strip()]

if st.button("🚀 Ejecutar Análisis"):
    with st.spinner("Sincronizando con mercados globales..."):
        df = analizar_seleccion(tickers_list)
        
        if not df.empty:
            # Cálculo de P/E Relativo
            try:
                mean_pe = df['P/E Fwd'].replace(0, np.nan).mean()
                df['P/E vs Sector'] = df['P/E Fwd'] / mean_pe
            except:
                df['P/E vs Sector'] = 1.0

            # Ordenamiento y TOP 10
            df_final = df.sort_values(by=["Rating Riesgo", "Alfa vs SPY" if "Alfa vs SPY" in df else "P/E Fwd"], 
                                      ascending=[True, True]).head(10)
            
            st.header("🏆 Selección TOP 10 de Research")
            
            # Mostrar tabla
            st.dataframe(df_final.style.format({
                "Precio": "${:.2f}",
                "P/E Fwd": "{:.2f}",
                "PEG Ratio": "{:.2f}",
                "P/B Ratio": "{:.2f}",
                "P/E vs Sector": "{:.2f}x",
                "Target P75": "${:.2f}",
                "VaR (%)": "{:.2f}%",
                "Entrada": "${:.2f}"
            }), use_container_width=True)

            # Gráficos
            col1, col2 = st.columns(2)
            with col1:
                fig_risk = go.Figure(go.Scatter(x=df_final["VaR (%)"], y=df_final["Target P75"], 
                                               mode='markers+text', text=df_final["Ticker"],
                                               marker=dict(size=12, color=df_final["Rating Riesgo"], colorscale='Viridis')))
                fig_risk.update_layout(title="Riesgo (VaR) vs Retorno (P75)", template="plotly_dark")
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col2:
                fig_val = go.Figure(go.Scatter(x=df_final["PEG Ratio"], y=df_final["P/B Ratio"], 
                                              mode='markers+text', text=df_final["Ticker"],
                                              marker=dict(size=12, color=df_final["P/E vs Sector"], colorscale='Cividis')))
                fig_val.update_layout(title="Valuación: PEG vs P/B", template="plotly_dark")
                st.plotly_chart(fig_val, use_container_width=True)
        else:
            st.error("No se pudieron obtener datos. Verifique su conexión o los tickers ingresados.")
