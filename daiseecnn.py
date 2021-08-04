import sys
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import time

class DaiseeCNN:
    def __init__(self):
        self.model = load_model('Xception_on_DAiSEE_fc.h5')

    def prediction(self, frame):
        frame = cv2.resize(frame, (299, 299))
        frame = np.expand_dims(frame, axis=0)
        frame = frame / 255

        predictions = self.model.predict(frame)
        # for prediction in predictions:
        #     print(prediction)

        #Engagement Labeling 중 가장 높은 확률의 정도를 return
        return np.argmax(predictions[1][0])