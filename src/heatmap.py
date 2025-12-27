import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- RUTAS DINÁMICAS (FIX) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'dataset_master_final.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No encuentro el dataset en: {DATA_PATH}")

# Carga de datos
df = pd.read_csv(DATA_PATH)

# Renombrar para que el gráfico se entienda bien
df = df.rename(columns={
    'housing_price_index': 'IPH', 
    'foreign_population': 'Inmigración (Stock)', 
    'unemployment_total': 'Paro', 
    'Euribor': 'Euribor', 
    'gdp_growth': 'PIB'
})

# Selección de variables para la correlación
cols = ['IPH', 'Inmigración (Stock)', 'Paro', 'Euribor', 'PIB']
corr_matrix = df[cols].corr()

# Generar Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Matriz de Correlación: Mercado Inmobiliario España (2007-2025)", fontsize=14)

# Guardar en carpeta results
output_path = os.path.join(RESULTS_DIR, 'matriz_correlacion.png')
plt.tight_layout()
plt.savefig(output_path)
print(f"✅ Matriz de correlación guardada en: {output_path}")

# plt.show() # Descomenta si quieres verla al momento