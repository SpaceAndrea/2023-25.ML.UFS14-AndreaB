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
model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model/model')
print(f"######## La model dir è: {model_dir}")
model = models.load_model(f"{model_dir}/model.keras")

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

@app.route('/ping', methods=['GET'])
def ping():
    """
    Endpoint per verificare lo stato del contenitore.
    SageMaker si aspetta una risposta 200 per il controllo dello stato.
    """
    try:
        # Aggiungi un semplice controllo sul modello per verificare che sia caricato
        if model:
            return "OK", 200
        else:
            return "Modello non caricato", 500
    except Exception as e:
        return str(e), 500

@app.route('/invocations', methods=['POST'])
def invocations():
    """
    Endpoint per l'inferenza.
    Si aspetta un'immagine inviata come file.
    """
    if 'file' not in request.files:
        return "Nessun file caricato", 400
    #file = request.files['file']
    #image_bytes = file.read()
    image_string_base_64 = request.get_data(as_text: true)
    image_bytes = # trovare il modo di trasformare da base64 a bytearray
    x = preprocess_image(image_bytes)
    
    # Predizione
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = labels[predicted_class[0]]
    
    return json.dumps({
        "valore": predicted_label
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
