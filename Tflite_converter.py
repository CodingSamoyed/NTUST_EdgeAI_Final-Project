import os
import numpy as np
import tensorflow as tf

keras_model = tf.keras.models.load_model('my_model')
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.experimental_new_converter = True
tflite_model = converter.convert()
# open('converted_model.tflite', 'wb').write(tflite_model)
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)