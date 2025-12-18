import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# --- CONFIGURACIÓN DE RUTAS ---
# 1. Obtenemos la ruta del script actual (dentro de src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Subimos un nivel para llegar a la raíz del proyecto (PROYECTO/)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 3. Definimos las rutas a los datos y resultados desde la raíz
DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Crear carpeta de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

FILE_PATH = os.path.join(DATA_DIR, 'dataset_master_final.csv')

if not os.path.exists(FILE_PATH):
    # Fallback por si ejecutas desde una ruta distinta
    if os.path.exists('dataset_master_final.csv'):
        FILE_PATH = 'dataset_master_final.csv'
    else:
        raise FileNotFoundError(f"No se encuentra {FILE_PATH}")

# --- 1. CARGA DE DATOS ---
df = pd.read_csv(FILE_PATH)

col_map = {
    'year': 'Año', 'Year': 'Año',
    'housing_price_index': 'IPH', 'Housing_Price_Index': 'IPH',
    'foreign_population': 'Poblacion_Extranjera', 'Foreign_Population': 'Poblacion_Extranjera',
    'unemployment_total': 'Paro', 'Unemployment_Total': 'Paro',
    'gdp_growth': 'PIB_Crecimiento',
    'Euribor': 'Euribor'
}
df = df.rename(columns=col_map)
df = df.sort_values('Año').ffill().bfill()

# --- 2. MODELADO ---
features = ['Poblacion_Extranjera', 'Paro', 'PIB_Crecimiento', 'Euribor']
X = df[features]
y = df['IPH']
weights = np.where(df['Año'] >= 2015, 3.0, 1.0)

model = make_pipeline(StandardScaler(), Ridge(alpha=0.5))
model.fit(X, y, ridge__sample_weight=weights)

# Score
r2 = r2_score(y, model.predict(X), sample_weight=weights)
print(f"\n✅ Modelo entrenado. R² Score: {r2:.4f}")

# --- 3. CALIBRACIÓN ---
last_row = df.iloc[-1]
last_year = int(last_row['Año'])
real_iph_2025 = last_row['IPH']
pred_2025 = model.predict(pd.DataFrame([last_row[features]], columns=features))[0]
offset = real_iph_2025 - pred_2025

# --- 4. SIMULACIÓN ---
future_years = [2026, 2027, 2028, 2029, 2030]
euribor_projection = [2.2, 2.1, 2.0, 2.0, 2.0] 

scenarios_config = {
    'A': {
        'label': 'A: Tendencia Actual (Boom)', 'color': 'red',
        'flow_trend': [600000, 650000, 700000, 750000, 800000],
        'paro_change': [-0.5, -0.5, -0.4, -0.4, -0.3], 'pib_val': 2.5 
    },
    'B': {
        'label': 'B: Frenazo y Ajuste (Caída)', 'color': 'green',
        'flow_trend': [100000, 50000, 0, -50000, -50000], 
        'paro_change': [0.5, 0.8, 0.8, 0.5, 0.5], 'pib_val': 0.5 
    },
    'C': {
        'label': 'C: Estabilización', 'color': 'orange',
        'flow_trend': [350000, 350000, 350000, 350000, 350000],
        'paro_change': [-0.1, 0.0, 0.0, 0.0, 0.0], 'pib_val': 1.5
    }
}

sim_results = {}
export_data = []

for key, config in scenarios_config.items():
    sim_years = [last_year]
    sim_iph = [real_iph_2025]
    current_pop = last_row['Poblacion_Extranjera']
    current_paro = last_row['Paro']
    
    # Dato base 2025 para Excel
    export_data.append({
        'Escenario': config['label'], 'Año': last_year, 'IPH': round(real_iph_2025, 2),
        'Poblacion_Ext': int(current_pop), 'Paro': round(current_paro, 2), 'Euribor': last_row['Euribor']
    })
    
    for i, year in enumerate(future_years):
        current_pop += config['flow_trend'][i]
        current_paro += config['paro_change'][i]
        current_paro = max(4.0, current_paro)
        current_euribor = euribor_projection[i]
        
        input_data = pd.DataFrame({
            'Poblacion_Extranjera': [current_pop], 'Paro': [current_paro],
            'PIB_Crecimiento': [config['pib_val']], 'Euribor': [current_euribor]
        })
        
        pred = model.predict(input_data)[0] + offset
        sim_years.append(year)
        sim_iph.append(pred)
        
        export_data.append({
            'Escenario': config['label'], 'Año': year, 'IPH': round(pred, 2),
            'Poblacion_Ext': int(current_pop), 'Paro': round(current_paro, 2), 'Euribor': current_euribor
        })
        
    sim_results[key] = {'years': sim_years, 'iph': sim_iph, 'info': config}

# --- 5. GUARDADO DE RESULTADOS ---
# A. Guardar CSV
csv_path = os.path.join(RESULTS_DIR, 'datos_simulacion.csv')
pd.DataFrame(export_data).to_csv(csv_path, index=False)
print(f"✅ Tabla guardada en: {csv_path}")

# B. Guardar Gráfico
plt.figure(figsize=(12, 7))
plt.plot(df['Año'], df['IPH'], 'k-', linewidth=3, label='Histórico')

for key, res in sim_results.items():
    plt.plot(res['years'], res['iph'], 
             color=res['info']['color'], linestyle='--', marker='o', linewidth=2.5,
             label=res['info']['label'])
    plt.text(2030.1, res['iph'][-1], f"{res['iph'][-1]:.1f}", 
             color=res['info']['color'], fontweight='bold', va='center')

plt.title(f'Proyección IPH 2026-2030 (R²: {r2:.2f}) - {os.path.basename(RESULTS_DIR)}', fontsize=14)
plt.ylabel('Índice de Precios de Vivienda (IPH)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.axvline(x=last_year, color='gray', linestyle=':', alpha=0.8)
plt.tight_layout()

img_path = os.path.join(RESULTS_DIR, 'grafica_proyeccion.png')
plt.savefig(img_path)
print(f"✅ Gráfico guardado en: {img_path}")
plt.show()