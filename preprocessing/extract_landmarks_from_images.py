import cv2
import mediapipe as mp
import os
import csv

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Carpetas de entrada y salida
DATASET_DIR = "dataset/asl_alphabet_processed/user_custom"
LANDMARKS_DIR = "dataset/landmarks"
os.makedirs(LANDMARKS_DIR, exist_ok=True)

# Archivo CSV para guardar los landmarks
csv_path = os.path.join(LANDMARKS_DIR, "hand_landmarks.csv")

# Crear el archivo CSV si no existe
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        header = ["label"] + [f"x_{i}" for i in range(21)] + [f"y_{i}" for i in range(21)] + [f"z_{i}" for i in range(21)]
        writer.writerow(header)

print("✅ Extrayendo puntos clave (landmarks) de las imágenes...")

for letter in sorted(os.listdir(DATASET_DIR)):
    letter_path = os.path.join(DATASET_DIR, letter)
    
    if not os.path.isdir(letter_path):
        continue
    
    print(f"✅ Procesando imágenes para '{letter}'...")

    for img_name in os.listdir(letter_path):
        img_path = os.path.join(letter_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ No se pudo leer {img_path}. Saltando...")
            continue

        # Convertir a RGB para MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        # Verificar si se detectó una mano
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Guardar puntos clave (landmarks) en CSV
                with open(csv_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    row = [letter]
                    for landmark in hand_landmarks.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z])
                    writer.writerow(row)
        else:
            print(f"⚠️ No se detectó mano en {img_path}. Imagen ignorada.")

hands.close()
print(f"✅ Extracción completada. Landmarks guardados en {csv_path}.")
