import cv2
import mediapipe as mp
import os
import json

# Obtener la ruta absoluta del directorio de dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "phrases_body")

# Verificar si el directorio existe
if not os.path.exists(DATASET_DIR):
    print("❌ No se encontró el directorio de frases capturadas.")
    exit()

print("✅ Iniciando extracción de landmarks para todas las frases...")

# Configuración de MediaPipe para cuerpo completo y manos
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Procesar cada carpeta de frases
for phrase in os.listdir(DATASET_DIR):
    phrase_dir = os.path.join(DATASET_DIR, phrase)
    if not os.path.isdir(phrase_dir):
        continue

    print(f"✅ Procesando frase: {phrase}...")

    # Crear carpeta de landmarks para la frase
    output_dir = os.path.join(phrase_dir, "landmarks")
    os.makedirs(output_dir, exist_ok=True)

    # Procesar cada imagen en la carpeta
    for img_name in os.listdir(phrase_dir):
        if not img_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(phrase_dir, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detectar puntos clave de cuerpo y manos
        pose_result = pose.process(image_rgb)
        hands_result = hands.process(image_rgb)

        landmarks_data = {
            "pose_landmarks": [],
            "hand_landmarks": []
        }

        # Extraer puntos clave de cuerpo completo
        if pose_result.pose_landmarks:
            for landmark in pose_result.pose_landmarks.landmark:
                landmarks_data["pose_landmarks"].append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })

        # Extraer puntos clave de las manos
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })
                landmarks_data["hand_landmarks"].append(hand_data)

        # Guardar puntos clave en archivo JSON
        json_filename = img_name.replace(".jpg", ".json")
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w") as json_file:
            json.dump(landmarks_data, json_file, indent=4)

        print(f"✅ Landmarks extraídos y guardados para {img_name}.")

print("✅ Extracción de landmarks completada para todas las frases.")
