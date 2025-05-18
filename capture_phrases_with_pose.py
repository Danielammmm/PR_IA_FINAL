import cv2
import mediapipe as mp
import os
import time

# Configuración de MediaPipe Pose (cuerpo completo) y Manos
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Configuración de la carpeta de frases
DATASET_DIR = "dataset/phrases_body"
os.makedirs(DATASET_DIR, exist_ok=True)

# Lista de frases disponibles
valid_phrases = [
    "como", "estas", "hola", "gracias", 
    "por_favor", "si", "no", 
    "buenos", "dias", "tardes", "noches"
]

print("✅ Escribe el nombre de la frase (sin espacios, usa '_'):")
print(f"Opciones disponibles: {', '.join(valid_phrases)}")
phrase = input("Frase: ").strip().lower().replace(" ", "_")

# Verificar si la frase es válida
if phrase not in valid_phrases:
    print(f"❌ Frase no reconocida. Usa solo una de estas: {valid_phrases}.")
    exit()

# Crear carpeta para la frase
phrase_dir = os.path.join(DATASET_DIR, phrase)
os.makedirs(phrase_dir, exist_ok=True)

# Iniciar la cámara
cap = cv2.VideoCapture(0)
print("✅ Cámara iniciada. Presiona 'ESPACIO' para comenzar la captura de la frase...")

capturing = False
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB para MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_pose = pose.process(image_rgb)
    result_hands = hands.process(image_rgb)

    # Dibujar puntos clave del cuerpo y manos
    if result_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar instrucciones
    cv2.putText(frame, f"Frase: {phrase} | Imgs: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Presiona 'ESPACIO' para iniciar la captura | 'ESC' para salir", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Iniciar captura con 'ESPACIO'
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # ESPACIO
        print("✅ Captura iniciando en 3 segundos...")
        time.sleep(3)  # Esperar 3 segundos
        capturing = True

    if capturing:
        # Guardar la imagen de la frase
        img_path = os.path.join(phrase_dir, f"{phrase}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        # Detener captura después de 30 imágenes (aprox 3 segundos)
        if count >= 30:
            print("✅ Captura completada.")
            capturing = False

    if key == 27:  # ESC para salir
        break

    # Mostrar la imagen
    cv2.imshow("Captura de Frases (Cuerpo Completo + Manos)", frame)

cap.release()
cv2.destroyAllWindows()
print(f"✅ Captura finalizada. {count} imágenes guardadas para la frase '{phrase}'.")

