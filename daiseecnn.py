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
        print(predictions[1])

        # #Engagement Labeling 중 가장 높은 확률의 정도를 return
        # return np.argmax(predictions[1][0])

        # # Heuristic하게 측정된 probability threshold를 이용
        # if predictions[1][0][0] > -4.0: #Engagement Label 0 변동성 처리
        #     return 0
        # elif predictions[1][0][1] > -2.0: #Engagement Label 1 변동성 처리
        #     return 1
        # else:
        #     return np.argmax(predictions[1][0])

        # (기준 low) Heuristic하게 측정된 probability threshold를 이용
        if predictions[1][0][0] > -4: #Engagement Label 0 변동성 처리
            return 0
        elif predictions[1][0][1] > -2: #Engagement Label 1 변동성 처리
            return 1
        else:
            return np.argmax(predictions[1][0])