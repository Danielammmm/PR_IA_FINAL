import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# Configuración de MediaPipe (Manos y Cuerpo)
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Definir las frases a capturar
phrases = [
    "buenos_dias", "buenas_tardes", "buenas_noches",
    "te_amo", "gracias", "de_nada", "por_favor", "hola_como_estas"
]

# Crear carpeta para las frases si no existe
dataset_dir = "dataset/phrases_video"
os.makedirs(dataset_dir, exist_ok=True)

print("✅ Frases disponibles:", phrases)
phrase = input("✅ Escribe la frase que deseas capturar (o 'exit' para salir): ").lower().replace(" ", "_")

if phrase not in phrases:
    print("❌ Frase no reconocida. Intenta nuevamente.")
    exit()

# Crear carpeta y archivo CSV para la frase
phrase_dir = os.path.join(dataset_dir, phrase)
os.makedirs(phrase_dir, exist_ok=True)
csv_file = os.path.join(phrase_dir, f"{phrase}.csv")

# Inicializar el CSV con encabezados
header = ["frame"] + [f"hand_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]] + \
         [f"pose_{i}_{axis}" for i in range(33) for axis in ["x", "y", "z"]]
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

# Iniciar la cámara
cap = cv2.VideoCapture(0)
print(f"✅ Grabando gestos para la frase: {phrase} (Presiona 'Espacio' para iniciar/parar)")

recording = False
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image_rgb)
    pose_results = pose.process(image_rgb)

    # Dibujar landmarks en la imagen
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.putText(frame, f"Frase: {phrase} (Espacio para grabar)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Verificar si se está grabando
    if recording:
        landmarks = [frame_count]

        # Obtener landmarks de las manos
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0] * (21 * 3))  # Si no hay mano, llenar con ceros

        # Obtener landmarks del cuerpo
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0] * (33 * 3))  # Si no hay cuerpo, llenar con ceros

        # Guardar en el CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(landmarks)
        
        frame_count += 1
        cv2.putText(frame, f"Grabando... ({frame_count} frames)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostrar la imagen
    cv2.imshow("Grabacion de Frases - Video", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC para salir
        break
    elif key == ord(" "):  # Espacio para iniciar/parar
        recording = not recording
        print(f"{'✅ Grabando' if recording else '⏸️ Detenido'}...")

cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()
print("✅ Grabacion completada.")
