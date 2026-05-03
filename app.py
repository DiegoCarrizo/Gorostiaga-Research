import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import skew, kurtosis

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gorostiaga Research | High-Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117 !important; color: #ffffff !important; }
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR CUANTITATIVO AVANZADO ---
@st.cache_data(ttl=3600)
def analizar_avanzado(tickers):
    resultados = []
    for t in tickers:
        try:
            asset = yf.Ticker(t)
            hist = yf.download(t, period="2y", progress=False, multi_level_index=False)
            if hist.empty: continue
            
            info = asset.info
            close = hist['Close']
            returns = np.log(close / close.shift(1)).dropna()
            
            # Métricas de Distribución (Momentos)
            s_val = skew(returns)
            k_val = kurtosis(returns) # Excess Kurtosis
            var_95 = np.percentile(returns, 5)
            
            # Monte Carlo (1000 rutas, 252 días)
            mu, sigma = returns.mean(), returns.std()
            last_p = float(close.iloc[-1])
            sims, days = 1000, 252
            yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
            paths = np.zeros_like(yields); paths[0] = last_p
            for i in range(1, days): paths[i] = paths[i-1] * yields[i]
            
            resultados.append({
                "Ticker": t,
                "Precio": last_p,
                "Sesgo": s_val,
                "Curtosis": k_val,
                "VaR 95%": var_95 * 100,
                "P75 Target": np.percentile(paths[-1], 75),
                "PEG": info.get('pegRatio', 0) or 0,
                "P/B": info.get('priceToBook', 0) or 0,
                "Sector": info.get('sector', 'N/A'),
                "Retornos": returns, # Para histograma
                "Paths": paths # Para gráfico Monte Carlo
            })
        except: continue
    return resultados

# --- INTERFAZ ---
st.title("🏛️ Gorostiaga Research - High-Quant Terminal")
tickers_input = st.sidebar.text_area("Lista de Tickers", "AAPL, MSFT, NVDA, GGAL, YPF, AL30.BA")
tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]

if st.button("🚀 Ejecutar Análisis Multidimensional"):
    data_list = analizar_avanzado(tickers)
    
    if data_list:
        df = pd.DataFrame(data_list).drop(['Retornos', 'Paths'], axis=1)
        st.header("🏆 Monitor de Valuación y Riesgo")
        st.dataframe(df.style.format({"Precio": "{:.2f}", "Sesgo": "{:.2f}", "Curtosis": "{:.2f}", "VaR 95%": "{:.2f}%", "P75 Target": "{:.2f}"}), use_container_width=True)

        # Seleccionar ticker para análisis profundo
        target = st.selectbox("Seleccione un activo para análisis de distribución y Monte Carlo", [d['Ticker'] for d in data_list])
        selected_data = next(item for item in data_list if item["Ticker"] == target)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 Distribución de Retornos: {target}")
            # Histograma + VaR Line
            res = selected_data['Retornos']
            fig_dist = ff.create_distplot([res], [target], bin_size=.005, show_curve=True, colors=['#00ffcc'])
            fig_dist.add_vline(x=selected_data['VaR 95%']/100, line_dash="dash", line_color="red", annotation_text="VaR 95%")
            fig_dist.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.info(f"**Sesgo:** {selected_data['Sesgo']:.2f} | **Curtosis:** {selected_data['Curtosis']:.2f}")
            st.caption("Un Sesgo negativo indica mayor probabilidad de caídas abruptas. Una Curtosis alta (>3) indica 'colas pesadas' (riesgo de eventos extremos).")

        with col2:
            st.subheader(f"🎲 Simulación Monte Carlo: {target}")
            paths = selected_data['Paths']
            fig_mc = go.Figure()
            for i in range(50): # Dibujar 50 rutas para claridad
                fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=0.5), opacity=0.3, showlegend=False))
            
            fig_mc.add_trace(go.Scatter(y=[selected_data['P75 Target']]*252, name="P75 Target", line=dict(color='#00ffcc', dash='dash')))
            fig_mc.update_layout(template="plotly_dark", yaxis_title="Precio Proyectado", xaxis_title="Ruedas (Días)")
            st.plotly_chart(fig_mc, use_container_width=True)

    else:
        st.error("No se pudieron procesar los datos.")
