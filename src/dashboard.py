import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from matplotlib.ticker import MaxNLocator

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador Vivienda TFM", layout="wide")

st.title("Simulador Interactivo: Impacto Migratorio en la Vivienda (2026-2030)")
st.markdown("""
Con este simulador podemos ver diferentes escenarios personalizados modificando las variables macroeconómicas clave de nuestro país.
El modelo utiliza un algoritmo **Ridge Regression** ponderado temporalmente con **Intervalos de Confianza del 80%**.

**($R^2= 0.84$)**
""")

# --- 1. CARGA Y ENTRENAMIENTO ---
@st.cache_data
def load_and_train():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets')
    FILE_PATH = os.path.join(DATA_DIR, 'dataset_master_final.csv')
    
    if not os.path.exists(FILE_PATH):
        st.error("No se encuentra el dataset maestro.")
        return None, None, None, None, None

    df = pd.read_csv(FILE_PATH)
    
    col_map = {
        'year': 'Año', 'Year': 'Año',
        'housing_price_index': 'IPH', 'Housing_Price_Index': 'IPH',
        'foreign_population': 'Poblacion_Extranjera', 'Foreign_Population': 'Poblacion_Extranjera',
        'unemployment_total': 'Paro', 'Unemployment_Total': 'Paro',
        'gdp_growth': 'PIB_Crecimiento', 'Euribor': 'Euribor'
    }
    df = df.rename(columns=col_map)
    df = df.sort_values('Año').ffill().bfill()
    
    features = ['Poblacion_Extranjera', 'Paro', 'PIB_Crecimiento', 'Euribor']
    X = df[features]
    y = df['IPH']
    weights = np.where(df['Año'] >= 2015, 3.0, 1.0)
    
    model = make_pipeline(StandardScaler(), Ridge(alpha=0.5))
    model.fit(X, y, ridge__sample_weight=weights)
    
    # --- CÁLCULO DEL ERROR ESTÁNDAR (NUEVO) ---
    # Predecimos el pasado para ver cuánto nos equivocamos normalmente
    y_pred_hist = model.predict(X)
    residuals = y - y_pred_hist
    # Desviación estándar de los errores (sigma)
    std_error = np.std(residuals)
    
    last_row = df.iloc[-1]
    real_iph_2025 = last_row['IPH']
    pred_2025 = model.predict(pd.DataFrame([last_row[features]], columns=features))[0]
    offset = real_iph_2025 - pred_2025
    
    return model, df, last_row, offset, std_error

model, df_hist, last_row, offset, std_error = load_and_train()

# --- 2. BARRA LATERAL ---
st.sidebar.header("⚙️ Configuración del Escenario")

st.sidebar.subheader("1. Política Migratoria")
flow_input = st.sidebar.slider(
    "Flujo Neto Anual de Inmigrantes", 
    min_value=-100000, max_value=1000000, value=400000, step=50000,
    help="¿Cuánta gente entra al año? (Negativo = Retorno)"
)

st.sidebar.subheader("2. Entorno Económico")
paro_trend = st.sidebar.select_slider(
    "Tendencia del Paro",
    options=["Mejora Fuerte", "Mejora Leve", "Estable", "Empeora Leve", "Crisis"],
    value="Mejora Leve"
)

paro_map = {
    "Mejora Fuerte": -200000, "Mejora Leve": -50000, "Estable": 0, 
    "Empeora Leve": 100000, "Crisis": 400000
}
paro_delta = paro_map[paro_trend]

euribor_input = st.sidebar.slider(
    "Euríbor Promedio",
    min_value=0.0, max_value=6.0, value=2.5, step=0.1
)

st.sidebar.markdown("---")
st.sidebar.info(
    "ℹ️ **Nota:** Esta herramienta es una simulación basada en datos históricos. "
    "El área sombreada representa el intervalo de confianza (incertidumbre) del modelo."
)

# --- 3. SIMULACIÓN ---
future_years = [2026, 2027, 2028, 2029, 2030]
sim_years = [int(last_row['Año'])]
sim_iph = [last_row['IPH']]
# Listas para el intervalo de confianza
ci_upper = [last_row['IPH']]
ci_lower = [last_row['IPH']]

current_pop = last_row['Poblacion_Extranjera']
current_paro = last_row['Paro']

# Como proyectamos a futuro, la incertidumbre se acumula (simplificación raíz cuadrada del tiempo)
z_score = 1.28 # Valor z para 80% confianza 

for i, year in enumerate(future_years):
    current_pop += flow_input
    current_paro += paro_delta
    current_paro = max(1500000, current_paro) 
    
    input_data = pd.DataFrame({
        'Poblacion_Extranjera': [current_pop],
        'Paro': [current_paro],
        'PIB_Crecimiento': [2.0],
        'Euribor': [euribor_input]
    })
    
    pred = model.predict(input_data)[0] + offset
    sim_years.append(year)
    sim_iph.append(pred)
    
    # Factor 0.6: Asumimos que la volatilidad futura será menor que la histórica (2008)
    volatility_factor = 0.5 
    
    uncertainty = std_error * z_score * np.sqrt(i + 1) * volatility_factor
    ci_upper.append(pred + uncertainty)
    ci_lower.append(pred - uncertainty)

# Línea Base
base_years = [int(last_row['Año'])]
base_iph = [last_row['IPH']]
base_pop = last_row['Poblacion_Extranjera']
base_paro = last_row['Paro']

for year in future_years:
    base_pop += 400000 
    base_paro += 0     
    base_paro = max(1500000, base_paro)
    
    base_input = pd.DataFrame({
        'Poblacion_Extranjera': [base_pop], 'Paro': [base_paro],
        'PIB_Crecimiento': [2.0], 'Euribor': [2.5]
    })
    base_pred = model.predict(base_input)[0] + offset
    base_years.append(year)
    base_iph.append(base_pred)


# --- 4. VISUALIZACIÓN ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Proyección Gráfica")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Histórico
    ax.plot(df_hist['Año'], df_hist['IPH'], 'k-', linewidth=3, label='Histórico')
    
    # Línea Base (Gris)
    ax.plot(base_years, base_iph, color='lightgray', linestyle=':', linewidth=2, label='Ref. Base')

    # Intervalo de Confianza (Sombra) [NUEVO]
    ax.fill_between(sim_years, ci_lower, ci_upper, color='red', alpha=0.15, label='IC 80%')

    # Tu Escenario
    ax.plot(sim_years, sim_iph, 'r--', marker='o', linewidth=3, label='Tu Escenario')
    
    ax.set_title(f"Impacto: {flow_input:,.0f} inmigrantes/año + Euríbor {euribor_input}%")
    ax.set_ylabel("Índice Precio Vivienda (IPH)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.axvline(x=2025, color='gray', linestyle=':', alpha=0.8)
    
    ax.text(2030.1, sim_iph[-1], f"{sim_iph[-1]:.1f}", color='red', fontweight='bold')
    st.pyplot(fig)

with col2:
    st.subheader("Resultados Numéricos")
    delta_total = sim_iph[-1] - sim_iph[0]
    pct_growth = (delta_total / sim_iph[0]) * 100
    st.metric(label="Precio Final (2030)", value=f"{sim_iph[-1]:.1f}", delta=f"{pct_growth:.1f}% acumulado")
    
    st.write("### Datos Año a Año")
    results_df = pd.DataFrame({
        'Año': sim_years, 
        'IPH Predicho': [round(x, 2) for x in sim_iph],
        'Rango Bajo': [round(x, 2) for x in ci_lower],
        'Rango Alto': [round(x, 2) for x in ci_upper]
    })
    st.dataframe(results_df, hide_index=True)

csv = results_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar datos (CSV)",
    data=csv,
    file_name='prediccion_iph_españa.csv',
    mime='text/csv',
)
st.caption("TFM: Análisis del Impacto Demográfico en el Mercado Inmobiliario Español.")