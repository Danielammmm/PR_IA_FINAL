# app.py
from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Modelo y clases
MODEL_PATH = "models/best_landmarks_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
class_names = [*list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "delete", "nothing", "space"]

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Leer la imagen enviada
    blob = request.files['frame'].read()
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res = hands.process(img_rgb)
    pred = "nothing"

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten().reshape(1, -1)
        out = model.predict(pts)
        pred = class_names[int(np.argmax(out))]

    # Codificar la imagen (JPEG + hex) para devolver landmarks
    _, buf = cv2.imencode('.jpg', img)
    return jsonify(prediction=pred, image=buf.tobytes().hex())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
