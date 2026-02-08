# python.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# 1️⃣ Cargar datos históricos
# -----------------------------
# Tu CSV debe tener columnas: mes, lat, lon, temp, humedad, enso, precipitacion
# Ejemplo:
# mes,lat,lon,temp,humedad,enso,precipitacion
# 2000-01,19.4,-99.1,22.5,70,0.2,55.1
df = pd.read_csv('datos.csv')
df = df.sort_values(by='mes').reset_index(drop=True)

# -----------------------------
# 2️⃣ Crear dataset con ventana de 3 meses
# -----------------------------
def crear_dataset(df, meses_atras=3):
    X, y = [], []
    for i in range(meses_atras, len(df)):
        # Variables históricas de los últimos 3 meses
        historial = df.iloc[i-meses_atras:i][['temp', 'humedad', 'enso', 'precipitacion']].values.flatten()
        # Coordenadas del punto actual
        coordenadas = df.iloc[i][['lat', 'lon']].values
        X.append(np.concatenate([historial, coordenadas]))
        y.append(df.iloc[i]['precipitacion'])
    return np.array(X), np.array(y)

X, y = crear_dataset(df)

# -----------------------------
# 3️⃣ Normalizar datos
# -----------------------------
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# -----------------------------
# 4️⃣ Crear modelo de red neuronal
# -----------------------------
model = Sequential([
    Dense(32, input_dim=X_scaled.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # Salida: precipitación
])

model.compile(optimizer='adam', loss='mse')

# -----------------------------
# 5️⃣ Entrenar modelo
# -----------------------------
model.fit(X_scaled, y_scaled, epochs=100, batch_size=16, validation_split=0.2)

# -----------------------------
# 6️⃣ Predecir precipitación para últimos 3 meses
# -----------------------------
# Aquí debes poner los valores de tus últimos 3 meses para temp, humedad, enso, precipitacion
# Y las coordenadas del punto donde quieres predecir
ultimos_3_meses = np.array([
    # Mes -3
    22.0, 65.0, 0.1, 50.0,
    # Mes -2
    22.5, 70.0, 0.2, 55.0,
    # Mes -1
    23.0, 68.0, 0.15, 60.0,
    # Coordenadas
    19.4, -99.1
])

ultimos_3_meses_scaled = scaler_X.transform(ultimos_3_meses.reshape(1, -1))
prediccion_scaled = model.predict(ultimos_3_meses_scaled)
prediccion = scaler_y.inverse_transform(prediccion_scaled)

print(f"Precipitación pronosticada: {prediccion[0][0]:.2f} mm")
