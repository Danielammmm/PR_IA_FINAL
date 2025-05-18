import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os

# Cargar el modelo entrenado
MODEL_PATH = "models/best_landmarks_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Cargar las clases (etiquetas de letras)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
               'Y', 'Z', 'delete', 'nothing', 'space']

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Iniciar la cámara
cap = cv2.VideoCapture(0)
print("✅ Cámara iniciada. Haz un signo con tu mano...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB para MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer puntos clave (landmarks)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Asegurarnos que los landmarks están completos (21 puntos)
            if len(landmarks) == 63:
                # Convertir a numpy array y predecir
                landmarks = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(landmarks)
                class_index = np.argmax(prediction)
                predicted_letter = class_names[class_index]
                confidence = np.max(prediction) * 100

                # Mostrar predicción en la imagen
                cv2.putText(frame, f"Letra: {predicted_letter} ({confidence:.2f}%)", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen
    cv2.imshow("Reconocimiento de Lenguaje de Señas - Landmarks", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("✅ Cámara cerrada.")
