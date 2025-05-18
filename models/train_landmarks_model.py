import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Cargar el CSV de Landmarks
CSV_PATH = "dataset/landmarks/hand_landmarks.csv"
data = pd.read_csv(CSV_PATH)

# Separar las etiquetas (label) de las coordenadas (X, Y, Z)
X = data.drop(columns=["label"]).values
y = data["label"].values

# Codificar las etiquetas (A-Z, delete, nothing, space)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
print("✅ Clases detectadas:", class_names)

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Red Neuronal (MLP)
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Configurar Early Stopping y Model Checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_landmarks_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, checkpoint]
)

print("✅ Modelo entrenado y guardado correctamente.")
