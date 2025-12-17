import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from functools import reduce

import statsmodels.api as sm
import os
from preprocessor import (
    procesar_iph, 
    procesar_stock_poblacion, 
    procesar_flujo_inmigracion,
    procesar_paro,
    procesar_pib,
    procesar_ipva,
    procesar_stock_vivienda_estatico
)

# --- CONFIGURACIÓN DE RUTAS ---
DATASET_PATH = r'C:\Users\Sergio\Desktop\Máster\proyecto\datasets'

# Nombres exactos de tus archivos
FILE_IPH = 'iph_bruto.csv'
FILE_POBLACION = 'poblacion_residente.csv'
FILE_FLUJO = 'flujo_inmigracion.csv'
FILE_PARO = 'paro.csv'
FILE_IPVA = 'indices IPVA.csv'
FILE_VIVIENDAS_STOCK = 'numero viviendas.csv'

# El PIB está en una subcarpeta, definimos su ruta relativa
FILE_PIB_PATH = os.path.join('pib', 'API_NY.GDP.MKTP.KD.ZG_DS2_es_csv_v2_331.csv')

try:
    print(">>> 1. INICIANDO PROCESAMIENTO DE DATASETS...\n")
    
    # --- A. SERIES TEMPORALES (Mergeables) ---
    
    # 1. Variable Objetivo (Precios General)
    df_iph = procesar_iph(os.path.join(DATASET_PATH, FILE_IPH))
    
    # 2. Variable Detallada (Precios por tipo)
    df_ipva = procesar_ipva(os.path.join(DATASET_PATH, FILE_IPVA))
    
    # 3. Variables Demográficas
    df_pob = procesar_stock_poblacion(os.path.join(DATASET_PATH, FILE_POBLACION))
    df_flow = procesar_flujo_inmigracion(os.path.join(DATASET_PATH, FILE_FLUJO))
    
    # 4. Variables Macroeconómicas
    df_paro = procesar_paro(os.path.join(DATASET_PATH, FILE_PARO))
    df_pib = procesar_pib(os.path.join(DATASET_PATH, FILE_PIB_PATH)) # Ojo a la ruta compuesta

    # --- B. DATOS ESTRUCTURALES (Contexto) ---
    # Solo lo cargamos para visualizarlo, no lo unimos al merge temporal
    stock_vivienda_2021 = procesar_stock_vivienda_estatico(os.path.join(DATASET_PATH, FILE_VIVIENDAS_STOCK))


    print("\n>>> 2. UNIFICANDO DATASETS EN UN 'DATASET MAESTRO'...")
    
    # Lista de DataFrames a unir
    dfs_to_merge = [df_iph, df_ipva, df_pob, df_flow, df_paro, df_pib]
    
    # Unimos todos usando 'year' como clave común
    df_master = reduce(lambda left, right: pd.merge(left, right, on='year', how='outer'), dfs_to_merge)
    
    # Ordenar por año
    df_master = df_master.sort_values('year').reset_index(drop=True)
    
    # Filtrar el rango de años útil para el análisis (ej. desde 2007)
    df_master = df_master[df_master['year'] >= 2007]

    print("\n✅ DATASET MAESTRO CREADO CON ÉXITO")
    print(f"Dimensiones: {df_master.shape}")
    print("\n--- Muestra de las últimas filas ---")
    print(df_master.tail(10))
    
    # Guardar resultado final
    output_file = os.path.join(DATASET_PATH, 'dataset_master_final.csv')
    df_master.to_csv(output_file, index=False)
    print(f"\n📁 Dataset listo guardado en: {output_file}")

except Exception as e:
    print(f"\n❌ ERROR CRÍTICO DURANTE LA EJECUCIÓN: {e}")
    # Esto ayuda a saber dónde falló exactamente si es un error de ruta
    import traceback
    traceback.print_exc()