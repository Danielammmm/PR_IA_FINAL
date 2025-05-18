import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuración
IMG_SIZE = (128, 128)
DATASET_DIR = "dataset/asl_alphabet_processed"
USER_DIR = os.path.join(DATASET_DIR, "user_custom")
EPOCHS = 20
BATCH_SIZE = 32

# Configurar Aumento de Datos (Data Augmentation)
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,
    fill_mode='nearest',
    validation_split=0.2  # 80% para entrenamiento, 20% para validación
)

# Crear generador de imágenes directamente desde disco
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    color_mode='rgb'  # Usar imágenes en color (RGB)
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    color_mode='rgb'
)

# Usar MobileNetV2 como base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # No entrenar la base

# Crear modelo mejorado
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Configurar Early Stopping y Model Checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_asl_letters_mobilenetv2.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Entrenar modelo con aumento de datos directamente desde disco
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

print("✅ Modelo MobileNetV2 entrenado y guardado correctamente.")
