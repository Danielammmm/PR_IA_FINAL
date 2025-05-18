import cv2
import mediapipe as mp
import pandas as pd
import os

# Configuración de MediaPipe para detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Directorio de videos
PHRASES_DIR = "dataset/phrases_video"
PHRASES_OUTPUT_DIR = "dataset/phrases_landmarks"
os.makedirs(PHRASES_OUTPUT_DIR, exist_ok=True)

phrases = [folder for folder in os.listdir(PHRASES_DIR) if os.path.isdir(os.path.join(PHRASES_DIR, folder))]

for phrase in phrases:
    print(f"✅ Procesando frase: {phrase}...")
    phrase_dir = os.path.join(PHRASES_DIR, phrase)
    phrase_output_dir = os.path.join(PHRASES_OUTPUT_DIR, phrase)
    os.makedirs(phrase_output_dir, exist_ok=True)
    
    for video_file in os.listdir(phrase_dir):
        if not video_file.endswith(".mp4"):
            continue
        
        video_path = os.path.join(phrase_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        landmarks_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir frame a RGB para MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks_data.append(landmarks)
            else:
                # Si no detecta mano, agrega una lista vacía
                landmarks_data.append([0] * 63)  # 21 puntos * 3 coordenadas

        cap.release()

        # Guardar los landmarks en un archivo CSV
        csv_filename = os.path.join(phrase_output_dir, f"{video_file.split('.')[0]}.csv")
        df = pd.DataFrame(landmarks_data)
        df.to_csv(csv_filename, index=False)

        print(f"✅ Landmarks guardados en: {csv_filename}")

hands.close()
print("✅ Todas las frases fueron procesadas.")
