# archivo.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

def mi_logica_python():
    try:
        # 1️⃣ Cargar datos históricos
        # Asegúrate de que 'datos.csv' esté en la misma carpeta que app.py
        if not os.path.exists('datos.csv'):
            return "Error: No se encontró el archivo datos.csv"

        df = pd.read_csv('datos.csv')
        df = df.sort_values(by='mes').reset_index(drop=True)

        # 2️⃣ Crear dataset
        def crear_dataset(df, meses_atras=3):
            X, y = [], []
            for i in range(meses_atras, len(df)):
                historial = df.iloc[i-meses_atras:i][['temp', 'humedad', 'enso', 'precipitacion']].values.flatten()
                coordenadas = df.iloc[i][['lat', 'lon']].values
                X.append(np.concatenate([historial, coordenadas]))
                y.append(df.iloc[i]['precipitacion'])
            return np.array(X), np.array(y)

        X, y = crear_dataset(df)

        # 3️⃣ Normalizar datos
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)

        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        # 4️⃣ Crear modelo
        model = Sequential([
            Dense(32, input_dim=X_scaled.shape[1], activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # 5️⃣ Entrenar modelo (He bajado las epochs para que la web no tarde tanto en responder)
        model.fit(X_scaled, y_scaled, epochs=50, batch_size=16, verbose=0)

        # 6️⃣ Predecir
        ultimos_3_meses = np.array([
            22.0, 65.0, 0.1, 50.0, # Mes -3
            22.5, 70.0, 0.2, 55.0, # Mes -2
            23.0, 68.0, 0.15, 60.0,# Mes -1
            19.4, -99.1            # Coordenadas
        ])

        ultimos_3_meses_scaled = scaler_X.transform(ultimos_3_meses.reshape(1, -1))
        prediccion_scaled = model.predict(ultimos_3_meses_scaled)
        prediccion = scaler_y.inverse_transform(prediccion_scaled)

        # ESTA ES LA RESPUESTA QUE IRÁ AL DIV
        resultado = f"Análisis completado. Precipitación pronosticada: {prediccion[0][0]:.2f} mm"
        return resultado

    except Exception as e:
        return f"Ocurrió un error en el código Python: {str(e)}"