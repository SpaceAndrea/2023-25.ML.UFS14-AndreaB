import logging
import json
import os
from flask import Flask, request
from keras import models
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import io

logging.debug('Init a Flask app')
app = Flask(__name__)

# Carica il modello e la mappatura delle classi
model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
print(f"######## La model dir è: {model_dir}")
model = models.load_model(f"{model_dir}/card_model.h5")

with open(f"{model_dir}/class_indices.pkl", 'rb') as f:
    class_indices = pickle.load(f)
labels = dict((v, k) for k, v in class_indices.items())

def preprocess_image(image_bytes):
    image_size = (224, 224)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(image_size)
    x = np.array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

@app.route('/ping', methods=['GET', 'POST'])
def ping():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Nessun file caricato", 400
        file = request.files['file']
        image_bytes = file.read()
        x = preprocess_image(image_bytes)
        
        # Predizione
        predictions = model.predict(x)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = labels[predicted_class[0]]
        
        # Estrai valore e seme
        valore, seme = predicted_label.split('_')
        
        return json.dumps({
            "valore": valore,
            "seme": seme
        })
    else:
        return "Per favore, invia un'immagine via POST."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
