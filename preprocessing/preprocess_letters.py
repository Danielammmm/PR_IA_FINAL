import os
import cv2
import numpy as np

# Ruta del dataset de entrenamiento
DATASET_DIR = "dataset/archive/asl_alphabet_train/asl_alphabet_train"
PROCESSED_DIR = "dataset/asl_alphabet_processed"

# Crear carpeta de salida
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Tamaño de imagen para el modelo
IMG_SIZE = (64, 64)

# Verificar nombres de carpetas especiales
SPECIAL_CLASSES = {
    "del": "delete",   # Para que reconozca "del" como "delete"
    "space": "space",
    "nothing": "nothing"
}

# Preprocesar imágenes
for letter_folder in os.listdir(DATASET_DIR):
    # Verificar si es una clase especial
    if letter_folder in SPECIAL_CLASSES:
        letter_folder_processed = SPECIAL_CLASSES[letter_folder]
    else:
        letter_folder_processed = letter_folder

    input_folder = os.path.join(DATASET_DIR, letter_folder)
    output_folder = os.path.join(PROCESSED_DIR, letter_folder_processed)
    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.equalizeHist(img)  # Mejorar contraste

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, img)

print("✅ Imágenes preprocesadas correctamente.")
