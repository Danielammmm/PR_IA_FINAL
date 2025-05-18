# README - Documentación del Proyecto de Traductor de Lenguaje de Señas

## Objetivo

Este proyecto permite desarrollar un sistema de traducción de lenguaje de señas en tiempo real, usando visión por computadora y aprendizaje profundo. La interfaz web permite capturar y traducir las señas realizadas con la mano, mostrando las letras detectadas y permitiendo al usuario construir frases.

## Instrucciones para Replicar el Proyecto

### 1. Configuración del Entorno

* Clona el repositorio del proyecto.
* Asegúrate de tener Python instalado (versión 3.8 o superior).
* Instala las dependencias del proyecto con el comando:

  ```bash
  pip install -r requirements.txt
  ```
* Verifica que las siguientes librerías estén correctamente instaladas:

  * TensorFlow
  * OpenCV
  * Flask
  * MediaPipe

### 2. Captura de Imágenes para Entrenamiento

* Usa la herramienta de captura en tiempo real para obtener imágenes de las señas.
* Asegúrate de capturar al menos 200 imágenes por letra (A-Z) y para las palabras especiales como "delete", "nothing" y "space".

### 3. Extracción de Puntos Clave (Landmarks)

* Ejecuta el script `extract_landmarks_from_images.py` para procesar las imágenes y extraer los puntos clave (landmarks).
* Esto generará un archivo CSV con los puntos clave listos para entrenamiento.

### 4. Entrenamiento del Modelo

* Ejecuta el script `train_landmarks_model.py` para entrenar el modelo.
* Asegúrate de que las imágenes están correctamente organizadas en carpetas por clase.
* El modelo se guardará automáticamente en la carpeta `models/`.

### 5. Configuración del Servidor Web

* El servidor web está basado en Flask.
* Ejecuta el script `app.py` para iniciar el servidor:

  ```bash
  python app.py
  ```
* Accede a la interfaz web en tu navegador en `http://localhost:5000`.

### 6. Uso del Traductor en Tiempo Real

* La cámara se activará automáticamente y comenzará a analizar las señas.
* La predicción se muestra en tiempo real, y puedes presionar el botón "Capturar Texto" para guardar las letras y formar palabras.

### 7. Solución de Problemas Comunes

* Si no detecta la cámara, verifica los permisos de tu navegador.
* Si las predicciones son incorrectas, asegúrate de que el modelo esté correctamente entrenado.
* Si hay problemas con Flask, verifica que estás usando el entorno virtual correcto (`venv`).
