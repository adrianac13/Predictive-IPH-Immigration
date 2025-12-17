import pandas as pd
import numpy as np
import os

# --- FUNCIONES AUXILIARES ---

def validar_transformacion(df_entrada, df_salida, nombre_dataset):
    """
    Imprime un informe de control de calidad.
    """
    print(f"\n--- REPORTE DE VALIDACIÓN: {nombre_dataset} ---")
    print(f"1. Filas Originales (aprox): {len(df_entrada)}")
    print(f"2. Filas Procesadas (Anuales): {len(df_salida)}")
    if not df_salida.empty:
        print(f"3. Rango Temporal: {df_salida['year'].min()} - {df_salida['year'].max()}")
        # Chequeo de Nulos
        nulos = df_salida.isnull().sum().sum()
        if nulos > 0:
            print(f"⚠️ ADVERTENCIA: Se encontraron {nulos} valores nulos en el dataset final.")
        else:
            print("✅ Integridad de datos: No hay valores nulos.")
    else:
        print("❌ ERROR: El dataset resultante está vacío.")
    print("-" * 40)

def limpiar_fecha_español(fecha_str):
    """Convierte fechas tipo '1 de enero de 2025' a datetime."""
    meses_map = {
        'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04', 
        'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08', 
        'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
    }
    if not isinstance(fecha_str, str): return np.nan
    parts = fecha_str.lower().split(' de ')
    if len(parts) != 3: return np.nan
    day, month_name, year = parts
    month = meses_map.get(month_name)
    if month: return f"{year}-{month}-{day.zfill(2)}"
    return np.nan

# --- FUNCIONES DE CARGA EXISTENTES ---

def procesar_iph(filepath):
    """
    Procesa el Índice de Precios de Vivienda.
    CORREGIDO: No elimina el primer año (2007) aunque no tenga crecimiento calculado.
    """
    df = pd.read_csv(filepath, sep=';', encoding='latin-1')
    df.columns = ['Total Nacional', 'CCAA', 'Tipo Vivienda', 'Medida', 'Periodo', 'Valor']
    df_clean = df[
        (df['Total Nacional'].str.contains('Nacional', na=False)) & 
        (df['Medida'].str.contains('ndice|Índice', na=False))
    ].copy()
    
    df_clean['Valor'] = df_clean['Valor'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df_clean['Valor'] = pd.to_numeric(df_clean['Valor'], errors='coerce')
    df_clean['year'] = df_clean['Periodo'].str[:4].astype(int)
    
    df_anual = df_clean.groupby('year')['Valor'].mean().reset_index()
    df_anual.rename(columns={'Valor': 'housing_price_index'}, inplace=True)
    df_anual['housing_price_growth'] = df_anual['housing_price_index'].pct_change() * 100
    
    # CORRECCIÓN: No hacemos dropna() aquí para no perder el dato de índice de 2007
    # df_final = df_anual.dropna() 
    df_final = df_anual.copy()
    
    validar_transformacion(df, df_final, "IPH (Vivienda)")
    return df_final

def procesar_flujo_inmigracion(filepath):
    """
    Procesa el Flujo de Inmigración.
    CORREGIDO: Lee 'Total' como string para evitar perder ceros finales (ej: 750.480 -> 750480).
    """
    # Forzamos que 'Total' se lea como texto
    df = pd.read_csv(filepath, sep=';', encoding='latin-1', dtype={'Total': str})
    
    df.columns = ['Sexo', 'Edad', 'Pais_Nacimiento', 'Nacionalidad', 'Pais_Origen', 'Periodo', 'Total']
    
    df_clean = df[
        (df['Pais_Origen'] == 'Total') & 
        (df['Sexo'] == 'Ambos sexos') &
        (df['Edad'] == 'Total') &
        (df['Pais_Nacimiento'] == 'Total') &
        (df['Nacionalidad'] == 'Total')
    ].copy()
    
    # Limpieza robusta: quitamos puntos de miles
    df_clean['Total'] = df_clean['Total'].str.replace('.', '', regex=False)
    df_clean['Total'] = pd.to_numeric(df_clean['Total'], errors='coerce')
    df_clean['year'] = df_clean['Periodo'].astype(int)
    
    df_final = df_clean[['year', 'Total']].rename(columns={'Total': 'immigration_flow'}).reset_index(drop=True)
    
    validar_transformacion(df, df_final, "Flujo Inmigración")
    return df_final

def procesar_stock_poblacion(filepath):
    """
    Procesa el Stock de Población.
    CORREGIDO: Lee 'Total' como string para precisión numérica.
    """
    df = pd.read_csv(filepath, sep=';', encoding='latin-1', dtype={'Total': str})
    df.columns = ['Pais_Nacimiento', 'Edad', 'Sexo', 'Periodo', 'Total']
    
    df_clean = df[(df['Edad'] == 'Todas las edades') & (df['Sexo'] == 'Total')].copy()
    
    # Limpieza robusta
    df_clean['Total'] = df_clean['Total'].str.replace('.', '', regex=False)
    df_clean['Total'] = pd.to_numeric(df_clean['Total'], errors='coerce')
    
    df_clean['Fecha'] = df_clean['Periodo'].apply(limpiar_fecha_español)
    df_clean['year'] = pd.to_datetime(df_clean['Fecha']).dt.year
    df_clean['Pais_Nacimiento'] = df_clean['Pais_Nacimiento'].str.replace('Espaa', 'España', regex=False)
    
    df_pivot = df_clean.pivot_table(index='year', columns='Pais_Nacimiento', values='Total', aggfunc='mean')
    
    if 'Total' in df_pivot.columns and 'España' in df_pivot.columns:
        df_pivot['foreign_population'] = df_pivot['Total'] - df_pivot['España']
        df_pivot['foreign_population_pct'] = (df_pivot['foreign_population'] / df_pivot['Total']) * 100
        df_final = df_pivot[['Total', 'foreign_population', 'foreign_population_pct']].reset_index()
        df_final.rename(columns={'Total': 'total_population'}, inplace=True)
        validar_transformacion(df, df_final, "Stock Población")
        return df_final
    else:
        raise ValueError("Columnas requeridas no encontradas en población.")

def procesar_paro(filepath):
    """
    Procesa el archivo de Paro (SEPE).
    Formato complejo: Encabezados en fila 5, separador coma, comillas en valores.
    """
    # 1. Leemos saltando las primeras 4 filas inútiles
    # header=0 significa que la fila 5 (índice 4 del original) es la cabecera
    df = pd.read_csv(filepath, sep=',', skiprows=4, encoding='latin-1')
    
    # 2. Limpieza de columnas (Las primeras están vacías en el csv original)
    # Renombramos manualmente las columnas relevantes basándonos en tu snippet
    # Columnas esperadas: [vacío, vacío, TOTAL, AGRICULTURA...]
    # Pero el CSV tiene índices de mes y año antes. Asignamos nombres genéricos primero.
    
    # Truco: Si las columnas no tienen nombre, Pandas las llama Unnamed.
    # Vamos a forzar el renombrado de las primeras columnas que sabemos qué son.
    nuevas_cols = list(df.columns)
    nuevas_cols[0] = 'year'
    nuevas_cols[2] = 'mes_nombre'
    nuevas_cols[3] = 'unemployment_total' # La columna "TOTAL"
    df.columns = nuevas_cols
    
    # 3. Filtrado y Limpieza de Valores
    # Eliminamos filas donde 'year' sea NaN (líneas vacías del CSV)
    df_clean = df.dropna(subset=['year']).copy()
    
    # Convertir 'year' a entero (puede venir como texto '2016')
    df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
    df_clean = df_clean.dropna(subset=['year']) # Eliminar si falló la conversión
    df_clean['year'] = df_clean['year'].astype(int)
    
    # Limpiar la columna de Paro Total
    # El formato viene como "4,150,755". Quitamos comas y convertimos.
    if df_clean['unemployment_total'].dtype == object:
        df_clean['unemployment_total'] = df_clean['unemployment_total'].astype(str).str.replace(',', '', regex=False).str.replace('.', '', regex=False)
    
    df_clean['unemployment_total'] = pd.to_numeric(df_clean['unemployment_total'], errors='coerce')
    
    # 4. Agregación Anual (Media de los 12 meses)
    df_anual = df_clean.groupby('year')['unemployment_total'].mean().reset_index()
    
    validar_transformacion(df, df_anual, "Paro Registrado (Media Anual)")
    return df_anual

def procesar_pib(filepath):
    """
    Procesa el archivo de PIB (Banco Mundial).
    Formato: Metadatos arriba, años en columnas.
    """
    # 1. Leemos saltando las 4 filas de metadatos
    df = pd.read_csv(filepath, sep=',', skiprows=4, encoding='utf-8')
    
    # 2. Filtramos solo España (ESP)
    df_esp = df[df['Country Code'] == 'ESP'].copy()
    
    if df_esp.empty:
        raise ValueError("No se encontró el código 'ESP' en el archivo de PIB.")
        
    # 3. Melt (Unpivot): Convertir columnas de años en filas
    # Identificamos las columnas que NO son años (Metadata)
    cols_meta = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    # Las columnas de años son el resto
    
    df_melted = df_esp.melt(id_vars=cols_meta, var_name='year', value_name='gdp_growth')
    
    # 4. Limpieza
    df_melted['year'] = pd.to_numeric(df_melted['year'], errors='coerce')
    df_melted = df_melted.dropna(subset=['year'])
    df_melted['year'] = df_melted['year'].astype(int)
    
    # El valor ya suele venir numérico en WB, pero por si acaso limpiamos
    df_melted['gdp_growth'] = pd.to_numeric(df_melted['gdp_growth'], errors='coerce')
    
    # Seleccionamos columnas finales
    df_final = df_melted[['year', 'gdp_growth']].dropna()
    
    validar_transformacion(df, df_final, "PIB (Crecimiento %)")
    return df_final

def procesar_ipva(filepath):
    """
    Procesa el IPVA (Índice de Precios de Vivienda) por tipo de edificación.
    CORREGIDO: Filtra explícitamente para excluir CCAA y evitar duplicados.
    """
    # Carga con latin-1 para las tildes
    df = pd.read_csv(filepath, sep=';', encoding='latin-1')
    
    # Renombrar columnas
    df.columns = ['Total Nacional', 'CCAA', 'Tipo Edificacion', 'Tipo Dato', 'Periodo', 'Total']
    
    # --- FILTRADO CRÍTICO ---
    # 1. CCAA debe ser NaN (vacío en el CSV) para asegurar que es dato NACIONAL.
    # 2. Tipo Dato debe ser Índice.
    df_clean = df[
        (df['CCAA'].isna()) & 
        (df['Tipo Dato'].str.contains('ndice|Índice', na=False))
    ].copy()
    
    # Limpieza de valores numéricos
    df_clean['Total'] = df_clean['Total'].astype(str).str.replace(',', '.', regex=False)
    df_clean['Total'] = pd.to_numeric(df_clean['Total'], errors='coerce')
    df_clean['year'] = df_clean['Periodo'].astype(int)
    
    # Pivotar: Ahora sí, cada combinación (year, Tipo Edificacion) es única
    df_pivot = df_clean.pivot(index='year', columns='Tipo Edificacion', values='Total').reset_index()
    
    # Renombrar columnas
    mapa_nombres = {
        'Total': 'ipva_total', 
        'Vivienda unifamiliar': 'ipva_unifamiliar',
        'Vivienda colectiva': 'ipva_colectiva'
    }
    df_pivot.rename(columns=mapa_nombres, inplace=True)
    
    # Seleccionar columnas finales
    # Aseguramos que existan las columnas antes de seleccionarlas (por si acaso falta alguna)
    cols_disponibles = [c for c in ['year', 'ipva_unifamiliar', 'ipva_colectiva'] if c in df_pivot.columns]
    df_final = df_pivot[cols_disponibles].copy()
    
    validar_transformacion(df, df_final, "IPVA (Unifamiliar/Colectiva)")
    return df_final

def procesar_stock_vivienda_estatico(filepath):
    """
    Extrae el DATOS ESTRUCTURAL del Censo (foto fija 2021).
    NO devuelve una serie temporal, sino un valor escalar o un pequeño DF informativo.
    """
    df = pd.read_csv(filepath, sep=';', encoding='latin-1')
    df.columns = ['Tipo Vivienda', 'Tamaño Municipio', 'Total']
    
    # Buscamos el "Total viviendas familiares" a nivel nacional ("Total (tamaño...)")
    # Filtramos por strings que contengan 'Total'
    dato_total = df[
        (df['Tipo Vivienda'].str.contains('Total viviendas familiares', na=False)) &
        (df['Tamaño Municipio'] == 'Total (tamaño de municipio)')
    ]
    
    if not dato_total.empty:
        valor_raw = dato_total['Total'].iloc[0]
        # Limpiar puntos de miles
        valor_limpio = int(str(valor_raw).replace('.', ''))
        
        print(f"\n--- DATO ESTRUCTURAL EXTRAÍDO (CENSO 2021) ---")
        print(f"Total Viviendas Familiares en España: {valor_limpio:,.0f}")
        print("NOTA: Este dato es estático y no se unirá al dataset temporal.")
        print("-" * 40)
        return valor_limpio
    else:
        print("⚠️ No se pudo extraer el stock total de viviendas.")
        return None