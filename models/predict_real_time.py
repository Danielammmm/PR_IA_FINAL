import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# Cargar el modelo entrenado
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "phrase_recognition_model.pkl")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "phrases_body")

print("✅ Cargando el modelo...")
model = joblib.load(MODEL_PATH)

# Configuración de MediaPipe (pose y manos)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Iniciar la cámara
cap = cv2.VideoCapture(0)
print("✅ Cámara iniciada. Presiona 'ESC' para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB para MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_pose = pose.process(image_rgb)
    result_hands = hands.process(image_rgb)

    # Dibujar puntos clave
    if result_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Extraer landmarks (cuerpo + manos)
    pose_landmarks = result_pose.pose_landmarks.landmark if result_pose.pose_landmarks else []
    hand_landmarks = [pt for hand in result_hands.multi_hand_landmarks for pt in hand.landmark] if result_hands.multi_hand_landmarks else []

    # Verificar si hay suficientes puntos clave
    if len(pose_landmarks) == 33 and len(hand_landmarks) >= 42:
        # Convertir landmarks a una lista de valores (x, y, z)
        pose_features = [coord for lm in pose_landmarks for coord in (lm.x, lm.y, lm.z)]
        hand_features = [coord for lm in hand_landmarks for coord in (lm.x, lm.y, lm.z)]

        # Rellenar manos si faltan puntos
        while len(hand_features) < 42 * 3:
            hand_features.extend([0, 0, 0])

        # Combinar características
        features = np.array(pose_features + hand_features).reshape(1, -1)

        # Realizar predicción
        prediction = model.predict(features)[0]
        cv2.putText(frame, f"Frase detectada: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Detectando cuerpo y manos...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar la imagen
    cv2.imshow("Reconocimiento de Frases (Tiempo Real)", frame)

    # Salir con 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
