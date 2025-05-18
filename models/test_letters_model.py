import cv2
import tensorflow as tf
import numpy as np
import os

# Cargar el modelo entrenado
MODEL_PATH = "models/best_asl_letters_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Cargar las clases (A-Z, delete, nothing, space)
DATASET_DIR = "dataset/asl_alphabet_processed"
class_names = sorted([label for label in os.listdir(DATASET_DIR) if label != "user_custom"])
print("✅ Modelo cargado y clases detectadas:", class_names)

# Configurar la cámara
cap = cv2.VideoCapture(0)
IMG_SIZE = (128, 128)

def preprocess_image(image):
    img = cv2.resize(image, (IMG_SIZE[0], IMG_SIZE[1]))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Ajustar para el modelo (batch)
    return img

print("✅ Presiona 'ESC' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar el área de detección
    x1, y1, x2, y2 = 200, 100, 450, 350
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    region_of_interest = frame[y1:y2, x1:x2]

    # Procesar la imagen capturada
    processed_image = preprocess_image(region_of_interest)
    prediction = model.predict(processed_image, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]

    # Mostrar la predicción
    cv2.putText(frame, f"Letra: {predicted_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Detección de Letras ASL", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Presionar ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
