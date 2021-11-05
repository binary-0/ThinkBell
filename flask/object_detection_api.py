# IMPORTS

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import json
from keras.models import load_model
from MediaPipeProcess import mediapipe_process
import daiseecnn


# if tf.__version__ != '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

# ENV SETUP  ### CWH: remove matplot display and manually add paths to references

# added to put object in JSON
class Object(object):
    def __init__(self):
        self.name="webrtcHacks TensorFlow Object Detection REST API"

    def toJSON(self):
        return json.dumps(self.__dict__)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def get_objects(image, threshold=0.5):
    # print(type(image))


    daisee = daiseecnn.DaiseeCNN()

    image_np = load_image_into_numpy_array(image)
    rgbFrame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

    daisee_prediction = daisee.prediction(image_np)

    print("예측 : ", daisee_prediction)

    EAR, headArea, handLM, poseLM = mediapipe_process(rgbFrame, rgbFrame, rgbFrame)

    print(f'EAR:{EAR} / headArea:{headArea}')

    # ouput={}
    # ouput['predict']=daisee_prediction
    # ouput['EAR']=EAR
    # ouput['headArea']=headArea
    # print(type(daisee_prediction), type(EAR), type(headArea))
    # outputJson = json.dumps({[1,2,3]})

    
    if not isinstance(daisee_prediction, int) :
        Dout=daisee_prediction.item()
    else:
        Dout=daisee_prediction

    if not isinstance(EAR, int):
        Eout=EAR.item()
    else:
        Eout=EAR

    outputJson = json.dumps({"predict":Dout, "EAR":Eout, "headarea":headArea})
    return outputJson
