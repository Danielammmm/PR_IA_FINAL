import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Obtener la ruta absoluta del dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "phrases_body")

# Función para cargar landmarks y etiquetas
def load_landmarks():
    X = []
    y = []

    # Verificar si el directorio existe
    if not os.path.exists(DATASET_DIR):
        print("❌ No se encontró el directorio de frases capturadas.")
        exit()

    # Recorrer las carpetas de frases
    for phrase in os.listdir(DATASET_DIR):
        phrase_dir = os.path.join(DATASET_DIR, phrase, "landmarks")
        if not os.path.exists(phrase_dir):
            continue

        # Cargar cada archivo JSON (landmarks)
        for json_file in os.listdir(phrase_dir):
            if not json_file.endswith(".json"):
                continue

            json_path = os.path.join(phrase_dir, json_file)
            with open(json_path, "r") as file:
                data = json.load(file)

            # Combinar landmarks de cuerpo y manos en una lista
            pose_landmarks = data.get("pose_landmarks", [])
            hand_landmarks = [pt for hand in data.get("hand_landmarks", []) for pt in hand]

            # Verificar si hay suficientes puntos clave (cuerpo + manos)
            if len(pose_landmarks) == 0:
                print(f"⚠️ Imagen ignorada (sin landmarks de cuerpo): {json_file}")
                continue

            # Asegurarnos de que las manos siempre tengan 42 puntos (21 por mano)
            while len(hand_landmarks) < 42:
                hand_landmarks.append({"x": 0, "y": 0, "z": 0})  # Rellenar con ceros

            # Convertir landmarks a una lista de valores (x, y, z)
            pose_features = [coord for lm in pose_landmarks for coord in (lm["x"], lm["y"], lm["z"])]
            hand_features = [coord for lm in hand_landmarks for coord in (lm["x"], lm["y"], lm["z"])]

            # Verificar que el tamaño sea consistente
            if len(pose_features) == 99 and len(hand_features) == 126:  # 33 x 3 (cuerpo), 42 x 3 (manos)
                features = pose_features + hand_features
                X.append(features)
                y.append(phrase)
            else:
                print(f"⚠️ Imagen ignorada (landmarks incompletos): {json_file}")

    return np.array(X, dtype=np.float32), np.array(y)

# Cargar los datos
print("✅ Cargando datos de landmarks...")
X, y = load_landmarks()
print(f"✅ Datos cargados: {len(X)} ejemplos.")

# Verificar si hay suficientes datos
if len(X) == 0:
    print("❌ No hay suficientes datos para entrenar el modelo.")
    exit()

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo RandomForest
print("✅ Entrenando el modelo...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
print("✅ Evaluando el modelo...")
y_pred = model.predict(X_test)
print("\n✅ Reporte de Clasificación:")
print(classification_report(y_test, y_pred))
print("\n✅ Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Guardar el modelo entrenado
MODEL_PATH = os.path.join(BASE_DIR, "models", "phrase_recognition_model.pkl")
joblib.dump(model, MODEL_PATH)
print(f"✅ Modelo guardado en: {MODEL_PATH}")
