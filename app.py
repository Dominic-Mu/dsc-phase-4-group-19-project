import os
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np
from PIL import Image

app = Flask(__name__)

class FixedDropout(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate, noise_shape=self.noise_shape, seed=self.seed)
        return inputs

    def get_config(self):
        config = super(FixedDropout, self).get_config()
        config.update({'rate': self.rate, 'noise_shape': self.noise_shape, 'seed': self.seed})
        return config


# Loading the model

model = load_model('Weather_Recognition_Model.h5', custom_objects={'FixedDropout': FixedDropout})


# Defining the classes for classification
weather_classes = ['cloudy', 'rain', 'shine', 'sunrise']
classes = weather_classes

# Image preprocessing function
def preprocess_image(image_path):
    """Function prepocessing image."""
    img = Image.open(image_path)
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# API endpoint for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    """Function predicting image weather class."""
    if request.method == 'POST':
        file = request.files['file']
        img_path = 'temp_image.jpg'
        file.save(img_path)

        processed_img = preprocess_image(img_path)

        # Making predictions using the model
        predictions = model.predict(processed_img)

        # Getting the predicted class
        predicted_class_index = np.argmax(predictions)
        predicted_class = classes[predicted_class_index]

        # Removing the temporary image file
        os.remove(img_path)

        # Returning the prediction
        return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
