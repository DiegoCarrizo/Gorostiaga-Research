import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import skew, kurtosis
import time

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="Gorostiaga Research | Terminal Quant V4", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117 !important; color: #ffffff !important; }
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    [data-testid="stTable"] { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR CUANTITATIVO ---
@st.cache_data(ttl=3600)
def analizar_universo_total(tickers):
    """Procesa el universo y calcula precios nominales para cada métrica."""
    analisis_completo = {}
    
    for t in tickers:
        try:
            # Descarga de datos
            df_hist = yf.download(t, period="2y", progress=False, multi_level_index=False)
            if df_hist.empty: continue
            
            asset = yf.Ticker(t)
            info = asset.info
            close = df_hist['Close']
            returns = np.log(close / close.shift(1)).dropna()
            
            # Estadísticos Base
            last_p = float(close.iloc[-1])
            s_val = float(skew(returns))
            k_val = float(kurtosis(returns))
            var_95_pct = float(np.percentile(returns, 5))
            
            # --- PRECIOS NOMINALES ---
            # El VaR en precio es el nivel que el activo podría tocar con un 5% de probabilidad en un día
            precio_var = last_p * np.exp(var_95_pct)
            
            # Monte Carlo (1000 rutas)
            mu, sigma = returns.mean(), returns.std()
            sims, days = 1000, 252
            yields = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((days, sims)))
            paths = np.zeros_like(yields); paths[0] = last_p
            for i in range(1, days):
                paths[i] = paths[i-1] * yields[i]
            
            p75_price = float(np.percentile(paths[-1], 75))

            analisis_completo[t] = {
                "Ticker": t,
                "Precio Actual": last_p,
                "Sesgo": s_val,
                "Curtosis": k_val,
                "VaR 95% (%)": var_95_pct * 100,
                "Precio VaR (S1)": precio_var, # Nivel de precio del VaR
                "Target P75 (Año)": p75_price,
                "Sector": info.get('sector', 'N/A'),
                "Retornos": returns.tolist(),
                "Paths": paths
            }
            time.sleep(0.1)
        except:
            continue
            
    return analisis_completo

# --- INTERFAZ ---
st.title("🏛️ Gorostiaga Research - Terminal de Análisis Multiactivo")

# Sidebar con tickers
tickers_input = st.sidebar.text_area("Universo de Tickers", "AAPL, MSFT, NVDA, GGAL, YPF, AL30.BA, TSLA, BABA")
tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]

if st.button("🚀 Ejecutar Monitor Completo"):
    with st.spinner("Sincronizando modelos de riesgo y precios nominales..."):
        full_data = analizar_universo_total(tickers)
        # Guardamos en el estado de la sesión para que persista al cambiar el selector
        st.session_state['full_data'] = full_data

# Verificar si hay datos cargados en la sesión
if 'full_data' in st.session_state and st.session_state['full_data']:
    data_dict = st.session_state['full_data']
    
    # 1. Tabla General de Referencia de Precios
    st.header("🏆 Monitor Global de Precios y Riesgo")
    df_resumen = pd.DataFrame([
        {k: v for k, v in v_data.items() if k not in ['Retornos', 'Paths']} 
        for v_data in data_dict.values()
    ])
    
    st.dataframe(df_resumen.style.format({
        "Precio Actual": "${:.2f}",
        "Sesgo": "{:.2f}",
        "Curtosis": "{:.2f}",
        "VaR 95% (%)": "{:.2f}%",
        "Precio VaR (S1)": "${:.2f}",
        "Target P75 (Año)": "${:.2f}"
    }), use_container_width=True)

    st.markdown("---")

    # 2. Selector con Sincronización Total
    target = st.selectbox("🎯 Análisis de Profundidad (Seleccione Ticker)", list(data_dict.keys()))
    d_target = data_dict[target]

    # Métricas destacadas en el encabezado del desglose
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio Actual", f"${d_target['Precio Actual']:.2f}")
    m2.metric("Nivel VaR (Soporte)", f"${d_target['Precio VaR (S1)']:.2f}", f"{d_target['VaR 95% (%)']:.2f}%", delta_color="inverse")
    m3.metric("Target P75 (1Y)", f"${d_target['Target P75 (Año)']:.2f}")
    m4.metric("Curtosis", f"{d_target['Curtosis']:.2f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📊 Distribución de Retornos: {target}")
        res_arr = np.array(d_target['Retornos'])
        fig_dist = ff.create_distplot([res_arr], [target], bin_size=.005, show_curve=True, colors=['#00ffcc'])
        
        # Línea de VaR en el gráfico
        fig_dist.add_vline(x=d_target['VaR 95% (%)']/100, line_dash="dash", line_color="red", 
                           annotation_text=f"VaR Price: ${d_target['Precio VaR (S1)']:.2f}")
        
        fig_dist.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.info(f"**Interpretación:** Para {target}, el sesgo de **{d_target['Sesgo']:.2f}** y la curtosis de **{d_target['Curtosis']:.2f}** sugieren un perfil de riesgo {'alto (Fat Tails)' if d_target['Curtosis'] > 3 else 'normal'}.")

    with col2:
        st.subheader(f"🎲 Rutas Monte Carlo: {target}")
        paths = d_target['Paths']
        fig_mc = go.Figure()
        
        for i in range(70): # Más rutas para visualización
            fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=0.6), opacity=0.2, showlegend=False))
        
        # Precio Objetivo Nominal
        fig_mc.add_trace(go.Scatter(y=[d_target['Target P75 (Año)']]*252, name=f"Target: ${d_target['Target P75 (Año)']:.2f}", 
                                   line=dict(color='#00ffcc', dash='dash', width=2)))
        
        # Precio VaR Nominal (como soporte base proyectado)
        fig_mc.add_trace(go.Scatter(y=[d_target['Precio VaR (S1)']]*252, name=f"Soporte VaR: ${d_target['Precio VaR (S1)']:.2f}", 
                                   line=dict(color='red', dash='dot', width=1)))
        
        fig_mc.update_layout(template="plotly_dark", yaxis_title="Precio Nominal (USD)", xaxis_title="Días Proyectados")
        st.plotly_chart(fig_mc, use_container_width=True)

else:
    st.info("💡 Ingrese sus tickers en la barra lateral y presione 'Ejecutar' para iniciar la terminal.")
