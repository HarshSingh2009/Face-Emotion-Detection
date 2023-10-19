import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image

class PredictionPipeline():
    def __init__(self) -> None:
        pass

    def predict_emotion(self, input_img):
        class_names = ['Happy', 'Sad']
        model = load_model('emotion_detection_model.h5')
        IMG_SIZE=256
        image = Image.open(input_img)
        resized_img = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        input_tensor = tf.cast(resized_img/255. , dtype=tf.float32)
        y_probs = model.predict(tf.expand_dims(input_tensor, axis=0))
        return tf.round(y_probs), y_probs