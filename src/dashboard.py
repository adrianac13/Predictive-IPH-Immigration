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
El modelo utiliza un algoritmo **Ridge Regression** ponderado temporalmente.

**($R^2= 0.84$)**
""")

# --- 1. CARGA Y ENTRENAMIENTO (Backend invisible) ---
@st.cache_data # Esto hace que no se recargue el modelo cada vez que tocas un botón
def load_and_train():
    # --- CONFIGURACIÓN DE RUTAS ---
    # 1. Obtenemos la ruta del script actual (dentro de src/)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 2. Subimos un nivel para llegar a la raíz del proyecto (PROYECTO/)
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    # 3. Definimos las rutas a los datos y resultados desde la raíz
    DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets')
    
    FILE_PATH = os.path.join(DATA_DIR, 'dataset_master_final.csv')
    
    if not os.path.exists(FILE_PATH):
        st.error("No se encuentra el dataset maestro.")
        return None, None, None, None

    df = pd.read_csv(FILE_PATH)
    
    # Limpieza
    col_map = {
        'year': 'Año', 'Year': 'Año',
        'housing_price_index': 'IPH', 'Housing_Price_Index': 'IPH',
        'foreign_population': 'Poblacion_Extranjera', 'Foreign_Population': 'Poblacion_Extranjera',
        'unemployment_total': 'Paro', 'Unemployment_Total': 'Paro',
        'gdp_growth': 'PIB_Crecimiento', 'Euribor': 'Euribor'
    }
    df = df.rename(columns=col_map)
    df = df.sort_values('Año').ffill().bfill()
    
    # Entrenamiento
    features = ['Poblacion_Extranjera', 'Paro', 'PIB_Crecimiento', 'Euribor']
    X = df[features]
    y = df['IPH']
    weights = np.where(df['Año'] >= 2015, 3.0, 1.0)
    
    model = make_pipeline(StandardScaler(), Ridge(alpha=0.5))
    model.fit(X, y, ridge__sample_weight=weights)
    
    # Datos base para proyectar
    last_row = df.iloc[-1]
    last_year = int(last_row['Año'])
    real_iph_2025 = last_row['IPH']
    
    # Calibración (Offset)
    pred_2025 = model.predict(pd.DataFrame([last_row[features]], columns=features))[0]
    offset = real_iph_2025 - pred_2025
    
    return model, df, last_row, offset

model, df_hist, last_row, offset = load_and_train()

# --- 2. BARRA LATERAL (CONTROLES) ---
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

# CORRECCIÓN DE ESCALA: Usamos número de personas, no porcentajes.
# Ejemplo: "Mejora Fuerte" = El paro baja en 200.000 personas cada año.
paro_map = {
    "Mejora Fuerte": -200000, 
    "Mejora Leve": -50000, 
    "Estable": 0, 
    "Empeora Leve": 100000, 
    "Crisis": 400000
}
paro_delta = paro_map[paro_trend]

euribor_input = st.sidebar.slider(
    "Euríbor Promedio",
    min_value=0.0, max_value=6.0, value=2.5, step=0.1
)

# --- NUEVO: DISCLAIMER ACADÉMICO ---
st.sidebar.markdown("---")
st.sidebar.info(
    "ℹ️ **Nota Académica:** Esta herramienta es una simulación basada en datos históricos. "
    "No constituye un asesoramiento financiero."
)

# --- 3. SIMULACIÓN EN TIEMPO REAL ---
future_years = [2026, 2027, 2028, 2029, 2030]
sim_years = [int(last_row['Año'])]
sim_iph = [last_row['IPH']]

current_pop = last_row['Poblacion_Extranjera']
current_paro = last_row['Paro']

for year in future_years:
    current_pop += flow_input
    current_paro += paro_delta
    
    # --- CORRECCIÓN SUELO DEL PARO ---
    # Mínimo 1.5 Millones de personas (Paro estructural), no 4.0
    current_paro = max(1500000, current_paro) 
    
    input_data = pd.DataFrame({
        'Poblacion_Extranjera': [current_pop],
        'Paro': [current_paro],
        'PIB_Crecimiento': [2.0], # Asumimos PIB constante para simplificar slider
        'Euribor': [euribor_input]
    })
    
    pred = model.predict(input_data)[0] + offset
    sim_years.append(year)
    sim_iph.append(pred)

# --- 4. VISUALIZACIÓN ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Proyección Gráfica")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Histórico
    ax.plot(df_hist['Año'], df_hist['IPH'], 'k-', linewidth=3, label='Histórico')
    
    # Proyección
    ax.plot(sim_years, sim_iph, 'r--', marker='o', linewidth=3, label='Tu Escenario')
    
    # --- MEJORA TOOLTIPS/TITULO ---
    # Usamos formato de miles para que se lea mejor (ej. 400k)
    ax.set_title(f"Impacto: {flow_input:,.0f} inmigrantes/año + Euríbor {euribor_input}%")
    ax.set_ylabel("Índice Precio Vivienda (IPH)")
    
    # --- MEJORA EJE X (Años enteros) ---
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axvline(x=2025, color='gray', linestyle=':')
    
    # Etiqueta final
    ax.text(2030.1, sim_iph[-1], f"{sim_iph[-1]:.1f}", color='red', fontweight='bold')
    
    st.pyplot(fig)

with col2:
    st.subheader("Resultados Numéricos")
    delta_total = sim_iph[-1] - sim_iph[0]
    pct_growth = (delta_total / sim_iph[0]) * 100
    
    st.metric(label="Precio Final (2030)", value=f"{sim_iph[-1]:.1f}", delta=f"{pct_growth:.1f}% acumulado")
    
    st.write("### Datos año a año")
    results_df = pd.DataFrame({'Año': sim_years, 'IPH Predicho': [round(x, 2) for x in sim_iph]})
    st.dataframe(results_df, hide_index=True)

# Nota al pie
st.caption("TFM: Análisis del Impacto Demográfico en el Mercado Inmobiliario Español.")