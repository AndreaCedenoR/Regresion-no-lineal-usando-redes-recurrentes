import pandas as pd
import matplotlib.pyplot as plt

# Cargar el CSV
df = pd.read_csv('data_numerica.csv')

# Eliminar columna innecesaria si existe
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Asegurarse de que la fecha esté en formato datetime
df['Date'] = pd.to_datetime(df['Date'])

# Renombrar la columna de valores para facilitar
df.rename(columns={'Monthly Mean Total Sunspot Number': 'Sunspots'}, inplace=True)

# Convertimos la columna de Sunspots a numérico por si hay errores
df['Sunspots'] = pd.to_numeric(df['Sunspots'], errors='coerce')

# Eliminar filas con datos faltantes, si hay
df.dropna(inplace=True)

# Graficar correctamente con fechas en el eje X
plt.figure(figsize=(12, 4))
plt.plot(df['Date'], df['Sunspots'], label='Sunspots', color='purple')
plt.title('Número mensual medio de manchas solares')
plt.xlabel('Fecha')
plt.ylabel('Manchas solares')
plt.grid(True)
plt.legend()
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split

# Convertimos la serie de manchas solares a un array numpy
data = df['Sunspots'].values

# Escogemos una longitud de ventana (número de pasos anteriores para predecir el siguiente)
window_size = 10

# Función para generar las secuencias
def create_sequences(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)

# Crear secuencias
X, y = create_sequences(data, window_size)

# Dividir en entrenamiento y validación (80% - 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

# Mostrar tamaños
print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

# Reestructurar para que Keras entienda que es una secuencia temporal
X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_rnn = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Definir el modelo
model_rnn = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(window_size, 1)),
    Dense(1)
])

# Compilar
model_rnn.compile(optimizer='adam', loss='mse')

# Resumen del modelo
model_rnn.summary()

# Entrenar el modelo
history_rnn = model_rnn.fit(
    X_train_rnn, y_train,
    epochs=300,
    batch_size=32,
    validation_data=(X_val_rnn, y_val)
)

import matplotlib.pyplot as plt

plt.plot(history_rnn.history['loss'], label='Train Loss')
plt.plot(history_rnn.history['val_loss'], label='Validation Loss')
plt.title('Pérdida (Loss) durante el entrenamiento - RNN')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

from tensorflow.keras.layers import LSTM

# Definir el modelo LSTM
model_lstm = Sequential([
    LSTM(50, activation='tanh', input_shape=(window_size, 1)),
    Dense(1)
])

# Compilar
model_lstm.compile(optimizer='adam', loss='mse')

# Resumen del modelo
model_lstm.summary()

# Entrenar el modelo LSTM
history_lstm = model_lstm.fit(
    X_train_rnn, y_train,
    epochs=300,
    batch_size=32,
    validation_data=(X_val_rnn, y_val)
)

plt.plot(history_lstm.history['loss'], label='Train Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.title('Pérdida (Loss) durante el entrenamiento - LSTM')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

# Predicciones
y_pred_rnn = model_rnn.predict(X_val_rnn)
y_pred_lstm = model_lstm.predict(X_val_rnn)

# Crear rango de fechas para el eje X (solo los datos de validación)
val_dates = df['Date'].values[-len(y_val):]

plt.figure(figsize=(14, 5))

# Gráfico de comparación
plt.plot(val_dates, y_val, label='Real', color='black')
plt.plot(val_dates, y_pred_rnn.flatten(), label='Predicción RNN', color='blue')
plt.plot(val_dates, y_pred_lstm.flatten(), label='Predicción LSTM', color='green')

plt.title('Comparación de predicciones: RNN vs LSTM')
plt.xlabel('Fecha')
plt.ylabel('Manchas solares')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_squared_error

# Cálculo del error cuadrático medio
mse_rnn = mean_squared_error(y_val, y_pred_rnn)
mse_lstm = mean_squared_error(y_val, y_pred_lstm)

print(f"MSE - RNN: {mse_rnn:.4f}")
print(f"MSE - LSTM: {mse_lstm:.4f}")
