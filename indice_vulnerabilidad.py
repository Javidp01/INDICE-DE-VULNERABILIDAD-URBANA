import pandas as pd
import numpy as np
import geopandas as gpd

# Ruta del archivo CSV
filePath_14 = r'C:\PRUEBAS\Tipo_14.csv'
df_14 = pd.read_csv(filePath_14, delimiter=",", low_memory=False)

# Me quedo solo con estas columnas de la tabla Tipo 14:
columns_to_keep_14 = ['31_pc', '79_aec', '84_stl', '105_tip']
df14_columns = df_14[columns_to_keep_14]
print(df14_columns.shape)
print("Número de columnas del dataset limpio: {0}".format(df14_columns.shape[1]))

# Se limpian todas las filas del DataFrame que son enteramente NaN (Not a Number)
df_14_cleanned = df14_columns.dropna(how='all').copy()
print("Número de filas del dataset limpio: {0}".format(df_14_cleanned.shape[0]))
print("Número de columnas del dataset limpio: {0}".format(df_14_cleanned.shape[1]))

# De la columna 105_tip (TIPOLOGIA CONSTRUCTIVA) nos quedamos con los 3 primeros números
df_14_cleanned['TIPOLOGIA_CON'] = df_14_cleanned['105_tip'].astype(str).str[0:3]
df_14_cleanned['CALIDAD_CON'] = df_14_cleanned['105_tip'].astype(str).str[3]

# Se renombran las columnas
df_final = df_14_cleanned.rename(columns={
    '31_pc': 'REF_CATASTRAL',
    '79_aec': 'ANT_VIV',
    '84_stl': 'SUPERF_M2'
})

# Asegurarse de que '105_tip' sea eliminado
df_final = df_final.drop('105_tip', axis=1)

# Ordenar las columnas correctamente
df_final = df_final[['REF_CATASTRAL', 'ANT_VIV', 'SUPERF_M2', 'CALIDAD_CON', 'TIPOLOGIA_CON']]

# Convertir tipos de datos
df_final['ANT_VIV'] = df_final['ANT_VIV'].astype(int)
df_final['SUPERF_M2'] = df_final['SUPERF_M2'].astype(float).round(2)

# Se filtran los valores de la columna TIPOLOGIA CONSTRUCTIVA
values = {'121', '122', '111', '112'}
df_final = df_final[df_final['TIPOLOGIA_CON'].isin(values)]

# Se agrupan las columnas por cada código de referencia catastral,
# realizando 'set' para saber si las columnas tienen más de un valor por fila y la 'media' de la superficie
df_filtered = df_final.groupby('REF_CATASTRAL').agg({
    "REF_CATASTRAL": 'unique',
    "ANT_VIV": lambda x: set(x),
    "SUPERF_M2": 'mean',
    "CALIDAD_CON": lambda x: set(x),
    "TIPOLOGIA_CON": lambda x: set(x)
})

# Se verifica si la columna ANTIGUEDAD VIVIENDA tiene más de un valor por fila
df_filtered["Nº elementos ANT_VIV"] = df_filtered["ANT_VIV"].apply(len)
if df_filtered['Nº elementos ANT_VIV'].nunique() > 1:
    print("La columna 'ANT_VIV' tiene más de un valor por fila.")
else:
    print("La columna 'ANT_VIV' tiene solo un valor por fila.")

# Se verifica si la columna CALIDAD CONSTRUCTIVA tiene más de un valor por fila
df_filtered["Nº elementos CALIDAD_CON"] = df_filtered["CALIDAD_CON"].apply(len)
if df_filtered['Nº elementos CALIDAD_CON'].nunique() > 1:
    print("La columna 'CALIDAD_CON' tiene más de un valor por fila.")
else:
    print("La columna 'CALIDAD_CON' tiene solo un valor por fila.")

# Se verifica si la columna TIPOLOGIA CONSTRUCTIVA tiene más de un valor por fila
df_filtered["Nº elementos TIPOLOGIA_CON"] = df_filtered["TIPOLOGIA_CON"].apply(len)
if df_filtered['Nº elementos TIPOLOGIA_CON'].nunique() > 1:
    print("La columna 'TIPOLOGIA_CON' tiene más de un valor por fila.")
else:
    print("La columna 'TIPOLOGIA_CON' tiene solo un valor por fila.")

# Se agrupan las columnas por cada código de referencia catastral,
# realizando la 'moda' de las columnas que tienen más de un valor por fila y la 'media' de la superficie
df_filtered = df_final.groupby('REF_CATASTRAL').agg({
    "REF_CATASTRAL": 'unique',
    "ANT_VIV": lambda x: list(pd.Series.mode(x))[0],
    "SUPERF_M2": 'mean',
    "CALIDAD_CON": lambda x: list(pd.Series.mode(x).astype(int))[0],  # Convertir a entero
    "TIPOLOGIA_CON": lambda x: list(pd.Series.mode(x))[0]
}).reset_index(drop=True)

# Se añaden las nuevas columnas para almacenar los nuevos valores del rango (de 1 a 4)
df_filtered['ANT_VALOR'] = np.nan
df_filtered['SUPERF_VAL'] = np.nan
df_filtered['CALIDAD_VAL'] = np.nan
df_filtered['TIPOLOGIA_VAL'] = np.nan

# Se define la función para los valores de antiguedad:
def get_value_antiguedad(año):
    if año <= 1955:
        valor = 4
    elif 1956 <= año <= 1975:
        valor = 3
    elif 1976 <= año <= 1995:
        valor = 2
    elif año >= 1996:
        valor = 1
    else:
        valor = 0
    return valor

# Se define la función para los valores de superficie:
def get_value_superficie(area):
    if area <= 60:
        valor = 4
    elif 60 < area <= 90:
        valor = 3
    elif 90 < area <= 120:
        valor = 2
    elif area > 120:
        valor = 1
    else:
        valor = 0
    return valor

# Se define la función para los valores de calidad constructiva
def get_value_calidad(valor):
    if valor in ['A', 'B', 'C']:
        return 4
    if 6 <= valor <= 9:
        return 4
    elif valor == 5:
        return 3
    elif valor == 4:
        return 2
    elif 1 <= valor <= 3:
        return 1
    else:
        return 0

# Se define la función para los valores de tipología constructiva
def get_value_tipologia(valor):
    if valor == '112':
        return 4
    elif valor == '111':
        return 3
    elif valor == '122':
        return 2
    elif valor == '121':
        return 1
    else:
        return 0

# Se rellena la columna ANT_VALOR:
df_filtered['ANT_VALOR'] = df_filtered['ANT_VIV'].apply(lambda x: get_value_antiguedad(x))

# Se rellena la columna SUPERF_VAL:
df_filtered['SUPERF_VAL'] = df_filtered['SUPERF_M2'].apply(lambda x: get_value_superficie(x))

# Se rellena la columna CALIDAD_VAL:
df_filtered['CALIDAD_VAL'] = df_filtered['CALIDAD_CON'].apply(lambda x: get_value_calidad(x))

# Se rellena la columna TIP_VALOR:
df_filtered['TIPOLOGIA_VAL'] = df_filtered['TIPOLOGIA_CON'].apply(lambda x: get_value_tipologia(x))

# Guardar el DataFrame final en un archivo CSV con comas como delimitador
output_file_path = r'C:\PRUEBAS\Tabla_tipo_14_getafe.csv'
df_filtered.to_csv(output_file_path, index=False, sep=',')
print(f"El DataFrame final se ha guardado en {output_file_path}")

# Leer el shapefile
shapefile_path = r'C:\PRUEBAS\PARCELA.SHP'
gdf = gpd.read_file(shapefile_path)

# Verificar las primeras filas del GeoDataFrame para ver las claves
print("GeoDataFrame antes de la unión:")
print(gdf.head())

# Leer el CSV
csv_path = r'C:\PRUEBAS\Tabla_tipo_14_getafe.csv'
df_csv = pd.read_csv(csv_path)

# Verificar las primeras filas del DataFrame para ver las claves
print("DataFrame CSV antes de la unión:")
print(df_csv.head())

# Asegurarnos de que las claves no tengan espacios en blanco y sean cadenas simples
gdf['REFCAT'] = gdf['REFCAT'].str.strip()
df_csv['REF_CATASTRAL'] = df_csv['REF_CATASTRAL'].apply(lambda x: x.strip("[]' "))

# Verificar que hay coincidencias entre las claves
common_keys = set(gdf['REFCAT']).intersection(set(df_csv['REF_CATASTRAL']))
print(f"Número de claves comunes: {len(common_keys)}")
if len(common_keys) == 0:
    print("No hay coincidencias entre las claves de unión. Verificar las claves.")

# Unir los DataFrames
gdf = gdf.merge(df_csv, how='left', left_on='REFCAT', right_on='REF_CATASTRAL')

# Verificar algunas filas después de la unión
print("GeoDataFrame después de la unión:")
print(gdf.head())

# Guardar el archivo sh-tante
output_shapefile_path = r'C:\PRUEBAS\PARCELA_UNIDA.SHP'
gdf.to_file(output_shapefile_path)
print(f"El shapefile unido se ha guardado en {output_shapefile_path}")