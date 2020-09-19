import tensorflow as tf
from keras.models import load_model
import cv2
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('model.h5')

img_test = cv2.imread('anh7.png', 0)
img_test = img_test.reshape(1, 28, 28, 1)
print(model.predict(img_test))