import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURACIÓN ---
DATASET_PATH = r'C:\Users\Sergio\Desktop\Máster\proyecto\datasets'
FILE_MASTER = 'dataset_master_final.csv'
MODEL_PATH = os.path.join(DATASET_PATH, 'best_housing_model.pkl')

# 1. Cargar Datos
df = pd.read_csv(os.path.join(DATASET_PATH, FILE_MASTER))

# Selección de Variables (FEATURES)
features = ['immigration_flow', 'foreign_population', 'total_population', 'unemployment_total', 'gdp_growth']
target = 'housing_price_index'

# --- CORRECCIÓN DE ERROR "Input X contains NaN" ---
# Antes de definir X e y, eliminamos cualquier fila que tenga huecos en las columnas clave.
# Esto eliminará probablemente el año 2007 si le falta la inmigración.
print(f"Filas antes de limpieza: {len(df)}")
df_clean = df.dropna(subset=features + [target]).copy()
print(f"Filas después de limpieza: {len(df_clean)}")

if len(df_clean) == 0:
    raise ValueError("¡ERROR CRÍTICO! Al limpiar NaNs te has quedado sin datos. Revisa el CSV.")

X = df_clean[features]
y = df_clean[target]

print(f"Dimensiones finales para el modelo: {X.shape}")

# 2. División Train / Test (Respetando el Tiempo)
# Cortamos manualmente: Entrenamos con el pasado (hasta 2021), Probamos con el presente (2022-2025)
cutoff_year = 2021
X_train = X[df_clean['year'] <= cutoff_year]
y_train = y[df_clean['year'] <= cutoff_year]
X_test = X[df_clean['year'] > cutoff_year]
y_test = y[df_clean['year'] > cutoff_year]

print(f"\nEntrenamiento: {len(X_train)} años (hasta {cutoff_year})")
print(f"Prueba (Validación): {len(X_test)} años ({cutoff_year+1}-2025)")

# 3. Escalado de Datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Importante: Transformamos el test con la media/desviación del train (para no hacer trampa)
X_test_scaled = scaler.transform(X_test)

# 4. Definición de Modelos a Competir
models = {
    "Regresión Lineal": LinearRegression(),
    "Ridge (Regularizado)": Ridge(alpha=1.0),
    # Ajustamos Random Forest para que no se sobreajuste con tan pocos datos
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
}

results = {}

print("\n--- RESULTADOS DEL TORNEO DE MODELOS ---")
print(f"{'Modelo':<25} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
print("-" * 65)

best_model = None
best_score = float('inf') # Buscamos el menor RMSE (Error Cuadrático Medio)
best_model_name = ""

for name, model in models.items():
    # Entrenar
    model.fit(X_train_scaled, y_train)
    
    # Predecir
    y_pred = model.predict(X_test_scaled)
    
    # Evaluar
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Model': model}
    
    print(f"{name:<25} | {mae:.2f}       | {rmse:.2f}       | {r2:.4f}")
    
    # Guardar el mejor
    if rmse < best_score:
        best_score = rmse
        best_model = model
        best_model_name = name

print("-" * 65)
print(f"\n🏆 GANADOR: {best_model_name} con un Error (RMSE) de {best_score:.2f}")
print("Nota: Un RMSE bajo indica que el modelo predice el índice de precios con mayor precisión.")

# 5. Guardar el modelo ganador y el scaler
final_pack = {
    'model': best_model,
    'scaler': scaler,
    'features': features,
    'model_name': best_model_name
}
joblib.dump(final_pack, MODEL_PATH)
print(f"Modelo guardado en: {MODEL_PATH}")

# 6. Visualización de Predicciones
plt.figure(figsize=(10, 6))

# Datos reales completos
plt.plot(df_clean['year'], df_clean['housing_price_index'], label='Datos Reales', color='black', linewidth=2)

# Predicción en Test (2022-2025)
years_test = df_clean[df_clean['year'] > cutoff_year]['year']
y_pred_best = best_model.predict(X_test_scaled)
plt.plot(years_test, y_pred_best, label=f'Predicción ({best_model_name})', color='red', linestyle='--', marker='o')

# Predicción en Train (para ver ajuste histórico)
years_train = df_clean[df_clean['year'] <= cutoff_year]['year']
y_train_pred = best_model.predict(X_train_scaled)
plt.plot(years_train, y_train_pred, label='Entrenamiento (Ajuste)', color='blue', alpha=0.3)

plt.axvline(x=cutoff_year + 0.5, color='gray', linestyle=':', label='Corte Train/Test')
plt.title(f'Evaluación del Modelo: Realidad vs Predicción ({best_model_name})', fontsize=14)
plt.xlabel('Año')
plt.ylabel('Índice Precio Vivienda')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 7. Importancia de Variables
plt.figure(figsize=(10, 5))

if hasattr(best_model, 'coef_'): # Lineales
    importance = best_model.coef_
    tipo_importancia = "Coeficientes (Peso)"
elif hasattr(best_model, 'feature_importances_'): # Árboles
    importance = best_model.feature_importances_
    tipo_importancia = "Importancia Relativa"

imp_df = pd.DataFrame({'Variable': features, 'Importancia': importance})
# Si son coeficientes, tomamos valor absoluto para ver magnitud
imp_df['Abs_Importancia'] = imp_df['Importancia'].abs()
imp_df = imp_df.sort_values(by='Abs_Importancia', ascending=False)

sns.barplot(x='Importancia', y='Variable', data=imp_df, palette='viridis')
plt.title(f'¿Qué mueve el precio de la vivienda?\nModelo: {best_model_name}')
plt.axvline(x=0, color='black', linewidth=1)
plt.show()