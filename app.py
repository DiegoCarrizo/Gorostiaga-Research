import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import skew, kurtosis
import time

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="Gorostiaga Research | Terminal Quant Global", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117 !important; color: #ffffff !important; }
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR CUANTITATIVO ---
@st.cache_data(ttl=3600)
def analizar_universo_completo(tickers):
    """Procesa todos los activos y guarda sus métricas estocásticas individuales."""
    todo_el_analisis = {}
    
    for t in tickers:
        try:
            # Descarga limpia para evitar errores de formato
            hist = yf.download(t, period="2y", progress=False, multi_level_index=False)
            if hist.empty: continue
            
            asset = yf.Ticker(t)
            info = asset.info
            close = hist['Close']
            returns = np.log(close / close.shift(1)).dropna()
            
            # Estadísticos de la Distribución
            s_val = float(skew(returns))
            k_val = float(kurtosis(returns))
            var_95 = float(np.percentile(returns, 5))
            
            # Simulación Monte Carlo (1000 rutas)
            mu, sigma = returns.mean(), returns.std()
            last_p = float(close.iloc[-1])
            sims, days = 1000, 252
            yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
            paths = np.zeros_like(yields); paths[0] = last_p
            for i in range(1, days):
                paths[i] = paths[i-1] * yields[i]
            
            # Guardamos el set completo de datos por cada Ticker
            todo_el_analisis[t] = {
                "Ticker": t,
                "Precio": last_p,
                "Sesgo": s_val,
                "Curtosis": k_val,
                "VaR 95%": var_95 * 100,
                "P75 Target": float(np.percentile(paths[-1], 75)),
                "Sector": info.get('sector', 'N/A'),
                "Retornos": returns.tolist(), # Convertimos a lista para estabilidad en el cache
                "Paths": paths # Array para gráficos
            }
            time.sleep(0.1) # Evitar rate limiting
        except Exception as e:
            continue
            
    return todo_el_analisis

# --- INTERFAZ PRINCIPAL ---
st.title("🏛️ Gorostiaga Research - Terminal de Análisis de Riesgo")

tickers_input = st.sidebar.text_area("Lista de Tickers", "AAPL, MSFT, NVDA, GGAL, YPF, AL30.BA")
tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]

if st.button("🚀 Ejecutar Análisis Multidimensional"):
    with st.spinner("Analizando distribución y simulando rutas estocásticas..."):
        resultados_dict = analizar_universo_completo(tickers)
    
    if resultados_dict:
        # 1. Tabla Resumen de todo el universo
        st.header("🏆 Monitor de Métricas Cuantitativas")
        df_resumen = pd.DataFrame([
            {k: v for k, v in data.items() if k not in ['Retornos', 'Paths']} 
            for data in resultados_dict.values()
        ])
        
        st.dataframe(df_resumen.style.format({
            "Precio": "{:.2f}", 
            "Sesgo": "{:.2f}", 
            "Curtosis": "{:.2f}", 
            "VaR 95%": "{:.2f}%", 
            "P75 Target": "{:.2f}"
        }), use_container_width=True)

        st.markdown("---")

        # 2. Selector de Activo para Análisis Profundo
        # Ahora el selector permite navegar por TODA la lista analizada
        target = st.selectbox("🎯 Seleccione un activo para desglosar Riesgo y Monte Carlo", list(resultados_dict.keys()))
        
        # Recuperamos la data específica del ticker seleccionado
        data_target = resultados_dict[target]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 Distribución de Retornos: {target}")
            res_arr = np.array(data_target['Retornos'])
            fig_dist = ff.create_distplot([res_arr], [target], bin_size=.005, show_curve=True, colors=['#00ffcc'])
            # Línea de VaR Crítico
            fig_dist.add_vline(x=data_target['VaR 95%']/100, line_dash="dash", line_color="red", 
                               annotation_text=f"VaR (95%): {data_target['VaR 95%']:.2f}%")
            fig_dist.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.info(f"**Sesgo:** {data_target['Sesgo']:.2f} | **Curtosis:** {data_target['Curtosis']:.2f}")

        with col2:
            st.subheader(f"🎲 Simulación Monte Carlo (1 año): {target}")
            paths = data_target['Paths']
            fig_mc = go.Figure()
            # Graficamos una muestra de 50 rutas para optimizar el renderizado
            for i in range(50):
                fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=0.6), opacity=0.3, showlegend=False))
            
            # Marcador de Precio Objetivo P75
            fig_mc.add_trace(go.Scatter(y=[data_target['P75 Target']]*252, name="P75 Target", 
                                       line=dict(color='#00ffcc', dash='dash', width=2)))
            fig_mc.update_layout(template="plotly_dark", yaxis_title="Precio (USD)", xaxis_title="Ruedas de Trading")
            st.plotly_chart(fig_mc, use_container_width=True)

    else:
        st.error("⚠️ No se pudieron obtener datos. Verifique la conexión con Yahoo Finance o los tickers ingresados.")
