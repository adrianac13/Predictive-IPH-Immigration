import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# --- CONFIGURACIÓN ---
FILE_PATH = 'datasets/dataset_master_final.csv'

if not os.path.exists(FILE_PATH):
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

# --- 2. SELECCIÓN DE VARIABLES ---
features = ['Poblacion_Extranjera', 'Paro', 'PIB_Crecimiento', 'Euribor']
X = df[features]
y = df['IPH']

# --- 3. PONDERACIÓN Y ENTRENAMIENTO ---
weights = np.where(df['Año'] >= 2015, 3.0, 1.0)
model = make_pipeline(StandardScaler(), Ridge(alpha=0.5))
model.fit(X, y, ridge__sample_weight=weights)

# --- 4. CÁLCULO DEL SCORE (LO QUE PEDÍAS) ---
# Predecimos sobre los datos históricos para ver qué tal lo hace
y_pred_hist = model.predict(X)

# R-Cuadrado (El % de varianza explicada. 1.0 es perfecto)
r2 = r2_score(y, y_pred_hist, sample_weight=weights)
# MAE (Error medio en puntos de índice)
mae = mean_absolute_error(y, y_pred_hist, sample_weight=weights)

print("\n" + "="*40)
print(f"   EVALUACIÓN DEL MODELO (SCORE)")
print("="*40)
print(f"✅ R² (Score de Ajuste): {r2:.4f} ({r2*100:.2f}%)")
print(f"📉 MAE (Error Medio):    {mae:.2f} puntos")
print("Interpreta esto: Tu modelo explica el {:.1f}% de la variación del precio.".format(r2*100))
print("="*40 + "\n")

# --- 5. CALIBRACIÓN ---
last_row = df.iloc[-1]
last_year = int(last_row['Año'])
real_iph_2025 = last_row['IPH']
pred_2025 = model.predict(pd.DataFrame([last_row[features]], columns=features))[0]
offset = real_iph_2025 - pred_2025

# --- 6. SIMULACIÓN DINÁMICA ---
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
export_data = [] # Lista para guardar datos y exportar a Excel

for key, config in scenarios_config.items():
    sim_years = [last_year]
    sim_iph = [real_iph_2025]
    
    current_pop_ext = last_row['Poblacion_Extranjera']
    current_paro = last_row['Paro']
    
    # Guardamos dato base 2025
    export_data.append({
        'Escenario': config['label'], 'Año': last_year, 'IPH_Predicho': real_iph_2025,
        'Poblacion_Extranjera': current_pop_ext, 'Paro': current_paro, 'Euribor': last_row['Euribor']
    })
    
    for i, year in enumerate(future_years):
        current_pop_ext += config['flow_trend'][i]
        current_paro += config['paro_change'][i]
        current_paro = max(4.0, current_paro)
        current_euribor = euribor_projection[i]
        
        input_data = pd.DataFrame({
            'Poblacion_Extranjera': [current_pop_ext], 'Paro': [current_paro],
            'PIB_Crecimiento': [config['pib_val']], 'Euribor': [current_euribor]
        })
        
        pred = model.predict(input_data)[0] + offset
        sim_years.append(year)
        sim_iph.append(pred)
        
        # Guardar para Excel
        export_data.append({
            'Escenario': config['label'], 'Año': year, 'IPH_Predicho': round(pred, 2),
            'Poblacion_Extranjera': current_pop_ext, 'Paro': round(current_paro, 2), 'Euribor': current_euribor
        })
        
    sim_results[key] = {'years': sim_years, 'iph': sim_iph, 'info': config}

# --- 7. EXPORTACIÓN A EXCEL/CSV ---
df_export = pd.DataFrame(export_data)
df_export.to_csv('datos_simulacion_tabla.csv', index=False)
print("✅ Tabla de datos exportada a 'datos_simulacion_tabla.csv'")

# --- 8. GRÁFICO ---
plt.figure(figsize=(12, 7))
plt.plot(df['Año'], df['IPH'], 'k-', linewidth=3, label='Histórico')

for key, res in sim_results.items():
    plt.plot(res['years'], res['iph'], 
             color=res['info']['color'], linestyle='--', marker='o', linewidth=2.5,
             label=res['info']['label'])
    final_val = res['iph'][-1]
    plt.text(2030.1, final_val, f"{final_val:.1f}", 
             color=res['info']['color'], fontweight='bold', va='center')

plt.title(f'Proyección IPH 2026-2030 (R² Score: {r2:.2f})', fontsize=14)
plt.ylabel('Índice de Precios de Vivienda (IPH)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.axvline(x=last_year, color='gray', linestyle=':', alpha=0.8)
plt.tight_layout()
plt.savefig('proyeccion_final_con_score.png')
print("✓ Gráfico generado: proyeccion_final_con_score.png")
plt.show()