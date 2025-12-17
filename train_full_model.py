import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURACIÓN ---
DATASET_PATH = r'C:\Users\Sergio\Desktop\Máster\proyecto\datasets'
FILE_MASTER = 'dataset_master_final.csv'
MODEL_PATH = os.path.join(DATASET_PATH, 'final_housing_model.pkl')

# 1. Cargar Datos
df = pd.read_csv(os.path.join(DATASET_PATH, FILE_MASTER))

# Selección de Variables (FEATURES)
features = ['immigration_flow', 'foreign_population', 'total_population', 'unemployment_total', 'gdp_growth']
target = 'housing_price_index'

# Limpieza de NaNs (vital para el año 2007 si falta inmigración)
df_clean = df.dropna(subset=features + [target]).copy()
X = df_clean[features]
y = df_clean[target]

print(f"Entrenando modelo final con {len(X)} años de datos (2008-2025).")

# 2. Escalado (Fit sobre TODO el dataset)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Selección del Modelo: Random Forest vs Ridge
# Vamos a entrenar ambos sobre todo el dataset y ver cuál se ajusta mejor a la curva histórica
model_rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model_ridge = Ridge(alpha=0.5)

model_rf.fit(X_scaled, y)
model_ridge.fit(X_scaled, y)

# Predecir sobre los mismos datos para ver el "Ajuste" (Fit)
y_pred_rf = model_rf.predict(X_scaled)
y_pred_ridge = model_ridge.predict(X_scaled)

rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
rmse_ridge = np.sqrt(mean_squared_error(y, y_pred_ridge))

print(f"\n--- AJUSTE HISTÓRICO (Errores sobre datos conocidos) ---")
print(f"Random Forest RMSE: {rmse_rf:.2f} (Capta no linealidad)")
print(f"Ridge Linear RMSE:  {rmse_ridge:.2f} (Capta tendencia global)")

# DECISIÓN AUTOMÁTICA
# Generalmente RF se ajustará mejor a los datos históricos, pero Ridge es mejor extrapolando tendencias.
# Para este proyecto, dada la complejidad reciente, elegiremos Random Forest si su error es bajo.
if rmse_rf < rmse_ridge:
    best_model = model_rf
    best_name = "Random Forest"
    print(f"\n✅ Seleccionado: {best_name}")
else:
    best_model = model_ridge
    best_name = "Ridge Regression"
    print(f"\n✅ Seleccionado: {best_name}")

# 4. Guardar el modelo definitivo
final_pack = {
    'model': best_model,
    'scaler': scaler,
    'features': features,
    'model_name': best_name
}
joblib.dump(final_pack, MODEL_PATH)
print(f"Modelo guardado en: {MODEL_PATH}")

# 5. Visualización Definitiva: Realidad vs Ajuste del Modelo
plt.figure(figsize=(12, 6))

# Datos Reales
plt.plot(df_clean['year'], y, label='Datos Reales', color='black', linewidth=3)

# Ajuste del Modelo
if best_name == "Random Forest":
    plt.plot(df_clean['year'], y_pred_rf, label='Ajuste Modelo (RF)', color='green', linestyle='--', marker='o')
else:
    plt.plot(df_clean['year'], y_pred_ridge, label='Ajuste Modelo (Ridge)', color='blue', linestyle='--', marker='o')

plt.title(f'Validación del Modelo Final: ¿Qué tan bien replica el pasado?\nModelo: {best_name}', fontsize=14)
plt.xlabel('Año')
plt.ylabel('Índice Precio Vivienda')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 6. Feature Importance Final
plt.figure(figsize=(10, 5))
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importance = np.abs(best_model.coef_)

imp_df = pd.DataFrame({'Variable': features, 'Importancia': importance})
imp_df = imp_df.sort_values(by='Importancia', ascending=False)

sns.barplot(x='Importancia', y='Variable', data=imp_df, palette='viridis')
plt.title(f'Factores Determinantes del Precio (Modelo Final)')
plt.show()