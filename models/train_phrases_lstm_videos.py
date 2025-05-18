import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping

# Directorio de los CSV de landmarks
PHRASES_DIR = "dataset/phrases_landmarks"
phrases = [folder for folder in os.listdir(PHRASES_DIR) if os.path.isdir(os.path.join(PHRASES_DIR, folder))]

print(f"✅ Frases detectadas para entrenamiento: {phrases}")

X = []
y = []

# Longitud fija para las secuencias
SEQUENCE_LENGTH = 50  # Ajustar a 50 frames por secuencia

# Cargar todos los CSV y generar secuencias
for phrase in phrases:
    phrase_dir = os.path.join(PHRASES_DIR, phrase)
    label_index = phrases.index(phrase)

    for csv_file in os.listdir(phrase_dir):
        if not csv_file.endswith(".csv"):
            continue

        csv_path = os.path.join(phrase_dir, csv_file)
        df = pd.read_csv(csv_path)
        landmarks = df.values

        # Crear ventanas de longitud fija
        for i in range(0, len(landmarks) - SEQUENCE_LENGTH, SEQUENCE_LENGTH // 2):
            sequence = landmarks[i:i + SEQUENCE_LENGTH]
            if len(sequence) == SEQUENCE_LENGTH:
                X.append(sequence)
                y.append(label_index)

# Convertir a numpy arrays
X = np.array(X)
y = np.array(y)

# Verificar que haya suficientes datos
if len(X) == 0:
    raise ValueError("❌ No se generaron secuencias. Verifica tus archivos CSV.")

print(f"✅ Total de secuencias: {len(X)}")

# Normalizar los valores (entre 0 y 1)
X = X / np.max(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo LSTM mejorado
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, X.shape[2])),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(len(phrases), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Usar EarlyStopping para evitar sobreentrenamiento
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=8,
    callbacks=[early_stopping]
)

# Guardar el modelo entrenado
model.save("models/best_phrases_lstm_videos.h5")
print("✅ Modelo LSTM para frases entrenado y guardado correctamente.")
