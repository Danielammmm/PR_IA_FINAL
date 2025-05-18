import cv2
import os

# Configuración
DATASET_DIR = "dataset/asl_alphabet_processed"
LABELS = [label.lower() for label in sorted(os.listdir(DATASET_DIR)) if label != "user_custom"]  # Ignorar user_custom

# Verificar si la carpeta de usuario existe y crearla
USER_DIR = os.path.join(DATASET_DIR, "user_custom")
os.makedirs(USER_DIR, exist_ok=True)

print("🔤 Letras disponibles:", LABELS)
print("✅ Escribe la letra que deseas capturar (o 'exit' para salir):")
label = input("Letra: ").strip().lower()

if label not in LABELS:
    print("❌ Letra no reconocida. Intenta nuevamente.")
    exit()

# Asegurarnos que se use la letra en mayúscula para guardar
label = label.capitalize() if len(label) == 1 else label

# Crear carpeta de la letra en user_custom si no existe
letter_dir = os.path.join(USER_DIR, label)
os.makedirs(letter_dir, exist_ok=True)

# Iniciar la cámara
cap = cv2.VideoCapture(0)
count = 0
capturing = False

print(f"📸 Presiona 'ESPACIO' para INICIAR la captura de '{label}'... 'ESC' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar área de captura (más cerca)
    x1, y1, x2, y2 = 250, 100, 500, 350
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Letra: {label} | Imgs: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Presiona ESPACIO para iniciar o detener", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Mostrar la imagen sin capturar
    cv2.imshow("Captura de Letras", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 32:  # ESPACIO para iniciar o detener captura
        capturing = not capturing
        if capturing:
            print("✅ Captura INICIADA. Presiona 'ESPACIO' nuevamente para detener.")
        else:
            print("❌ Captura DETENIDA. Presiona 'ESPACIO' para reanudar.")

    if capturing:
        # Capturar solo dentro del área
        region_of_interest = frame[y1:y2, x1:x2]
        region_resized = cv2.resize(region_of_interest, (128, 128))
        
        # Guardar imagen en user_custom
        img_path = os.path.join(letter_dir, f"{label}_{count}.jpg")
        cv2.imwrite(img_path, region_resized)
        count += 1

    if key == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Captura completada. {count} imágenes guardadas para '{label}'.")
