import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURACIÓN ---
# Estilo de gráficos profesional
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Cargar datos
DATASET_PATH = r'C:\Users\Sergio\Desktop\Máster\proyecto\datasets'
FILE_MASTER = 'dataset_master_final.csv'
df = pd.read_csv(os.path.join(DATASET_PATH, FILE_MASTER))

print(f"Dataset cargado. Filas: {len(df)}")

# --- 1. MATRIZ DE CORRELACIÓN (HEATMAP) ---
# Seleccionamos solo las variables numéricas clave para no ensuciar el gráfico
cols_interes = [
    'housing_price_index', 
    'housing_price_growth',
    'immigration_flow', 
    'foreign_population', 
    'total_population',
    'unemployment_total', 
    'gdp_growth'
]

plt.figure(figsize=(10, 8))
# Calculamos la correlación de Pearson
corr_matrix = df[cols_interes].corr()

# Generamos el Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Mapa de Calor: Correlaciones entre Inmigración, Economía y Vivienda', fontsize=16)
plt.tight_layout()
plt.show()

# --- 2. GRÁFICO DUAL: PRECIO VIVIENDA vs INMIGRACIÓN (La prueba visual) ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Eje izquierdo: Precio Vivienda
color = 'tab:red'
ax1.set_xlabel('Año', fontsize=12)
ax1.set_ylabel('Índice Precio Vivienda (IPH)', color=color, fontsize=12)
ax1.plot(df['year'], df['housing_price_index'], color=color, linewidth=3, label='Precio Vivienda (IPH)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(False) # Quitamos grid para limpiar

# Eje derecho: Flujo Inmigración
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Flujo de Inmigración (Personas/Año)', color=color, fontsize=12)
ax2.plot(df['year'], df['immigration_flow'], color=color, linewidth=3, linestyle='--', label='Flujo Inmigración')
ax2.tick_params(axis='y', labelcolor=color)

# Título y sombras de crisis/boom
plt.title('Relación Histórica: Precio de Vivienda vs Flujo Migratorio (2007-2025)', fontsize=16)
plt.axvspan(2008, 2013, color='gray', alpha=0.1, label='Crisis Financiera')
plt.axvspan(2021, 2025, color='green', alpha=0.1, label='Boom Post-COVID')

fig.tight_layout()
plt.show()

# --- 3. SCATTER PLOT: ¿MÁS INMIGRACIÓN = MÁS PRECIO? ---
# Regresión lineal simple visual
plt.figure(figsize=(10, 6))
sns.regplot(x='immigration_flow', y='housing_price_index', data=df, 
            scatter_kws={'s': 100, 'color': 'blue'}, line_kws={'color': 'red'})

# Etiquetar algunos años clave
for i in range(df.shape[0]):
    if df.year[i] in [2007, 2013, 2019, 2024]: # Años representativos
        plt.text(df.immigration_flow[i]+10000, df.housing_price_index[i], 
                 str(df.year[i]), fontsize=10, weight='bold')

plt.title('Dispersión: Impacto del Flujo Migratorio en el Índice de Precios', fontsize=14)
plt.xlabel('Flujo de Inmigración Anual')
plt.ylabel('Índice de Precios de Vivienda')
plt.show()

# --- 4. EVOLUCIÓN COMPARADA NORMALIZADA (BASE 100) ---
# Para ver qué crece más rápido, normalizamos todo a base 100 en el año 2015 (punto de inflexión)
df_norm = df.copy()
base_year = 2015
idx_base = df_norm.index[df_norm['year'] == base_year].tolist()[0]

for col in ['housing_price_index', 'total_population', 'foreign_population']:
    val_base = df_norm.at[idx_base, col]
    df_norm[col + '_base100'] = (df_norm[col] / val_base) * 100

plt.figure(figsize=(12, 6))
plt.plot(df_norm['year'], df_norm['housing_price_index_base100'], label='Precio Vivienda', linewidth=2, color='red')
plt.plot(df_norm['year'], df_norm['total_population_base100'], label='Población Total', linewidth=2, color='gray', linestyle=':')
plt.plot(df_norm['year'], df_norm['foreign_population_base100'], label='Población Extranjera (Stock)', linewidth=2, color='blue')

plt.axhline(100, color='black', linewidth=1)
plt.legend()
plt.title('Crecimiento Comparado (Base 100 = Año 2015)', fontsize=14)
plt.ylabel('Índice (2015=100)')
plt.show()