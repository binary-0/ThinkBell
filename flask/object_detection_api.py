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
import time
from queue import Queue


# if tf.__version__ != '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

# ENV SETUP  ### CWH: remove matplot display and manually add paths to references

class detectionQueue:
    def __init__(self):
        self.que = Queue()
        self.size = 0
        self.sum = 0
        self.avg = 0

        self.colorCnt = [0, 0, 0] #red, creem, green

        self.maxSize = 5

    def detectionPush(self, data):
        if data is 0 or data is 1:
            self.colorCnt[0] += 1
        elif data is 2:
            self.colorCnt[1] += 1
        else:
            self.colorCnt[2] += 1

        if self.size < self.maxSize:
            self.que.put(data)

            self.size += 1

            return self.colorCnt.index(max(self.colorCnt))
        else:
            deq = self.que.get()
            if deq is 0 or deq is 1:
                self.colorCnt[0] -= 1
            elif deq is 2:
                self.colorCnt[1] -= 1
            else:
                self.colorCnt[2] -= 1

            self.que.put(data)

            return self.colorCnt.index(max(self.colorCnt))

##############
##############
global daisee
global prevGestureIdx
global frCnt
global CALITIME
global caliEnd
global startTime
global prevTime
global TIMETHRESHOLD
global CNNTime
global headSum
global headAvg
global headPoseSum
global headPoseAvg
global EARSum
global EARAvg
global EARTime
global dQ
global firstImg
global predOnce
global faceAddedTime
global EARAddedTime
global curTime
global sec

global headStatus
global headPoseStatus
global handGestureStatus
global EARStatus

daisee = daiseecnn.DaiseeCNN()
prevGestureIdx = -1

frCnt = 0

CALITIME = 13
caliEnd = False

startTime = time.time()
prevTime = startTime
TIMETHRESHOLD = 3

CNNTime = None

headSum = 0
headAvg = None

headPoseSum = [0, 0]
headPoseAvg = [None, None]

EARSum = 0
EARAvg = None
EARTime = None

dQ = detectionQueue()

firstImg = cv2.imread('./FirstImage.jpg')
predOnce = False

faceAddedTime = 0
EARAddedTime = 0

curTime = time.time()
sec = curTime - prevTime
prevTime = curTime

headStatus = None
headPoseStatus = None
EARStatus = False
handGestureStatus = -1

ERROR_OUTPUT = json.dumps({
    'colorStat': '-1',
    'generalStat': '-1',
    'handStat': '-1'
})

############
############

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
    global daisee
    global prevGestureIdx
    global frCnt
    global CALITIME
    global caliEnd
    global startTime
    global prevTime
    global TIMETHRESHOLD
    global CNNTime
    global headSum
    global headAvg
    global headPoseSum
    global headPoseAvg
    global EARSum
    global EARAvg
    global EARTime
    global dQ
    global firstImg
    global predOnce
    global faceAddedTime
    global EARAddedTime
    global curTime
    global sec
    
    global headStatus
    global headPoseStatus
    global handGestureStatus
    global EARStatus

    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime

    if predOnce is False:
        print("predictMae")
        dummy = daisee.prediction(firstImg)
        predOnce = True
        startTime = time.time()
        return ERROR_OUTPUT #Send 오류코드

    image_np = load_image_into_numpy_array(image)
    frame = image_np
    if frame is None:
        return ERROR_OUTPUT #Send 오류코드
    
    frCnt += 1
    try:
        print('FPS: ' + str(1 / (sec)))
    except:
        print('FPS___')

    try:
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        print("Couldn't grapped. Maybe video is changing..")
        return ERROR_OUTPUT #Send 오류코드

    ### FOR YOLO ###
    
    handGestureTemp = -1
    
    EAR, headArea, headPose, handLM, poseLM, handGesIdx = mediapipe_process(rgbFrame, rgbFrame, rgbFrame)

    handGestureTemp = handGesIdx
    #print(f'prev:{prevGestureIdx[i]} and cur:{handGestureTemp}')
    if handGestureTemp is not -1 and prevGestureIdx == handGestureTemp:
        #print(f'//////////haittaa with {handGestureTemp}')
        if handGestureTemp == 5 or handGestureTemp == 9 or handGestureTemp == 10:
            handGestureStatus = handGestureTemp
        prevGestureIdx = handGestureTemp
    else:
        handGestureStatus = -1
        prevGestureIdx = handGestureTemp

    if caliEnd is False:
        if time.time() - startTime > CALITIME: # CALIBRATION DONE
            if faceAddedTime == 0 or EARAddedTime == 0:
                print('Cannot Completed Calibration: because of less count of addition')
                print('Program is going to recalibrate...')
                startTime = time.time()
            else:
                headAvg = headSum / faceAddedTime
                EARAvg = EARSum / EARAddedTime
                headPoseAvg[0] = headPoseSum[0] / faceAddedTime
                headPoseAvg[1] = headPoseSum[1] / faceAddedTime
                
                caliEnd = True
                print('////////////////////CALIEND//////////')
        else:
            if EAR is not -1 and headArea is not -1:
                headSum += headArea
                headPoseSum[0] += headPose[0]
                headPoseSum[1] += headPose[1]
                faceAddedTime += 1
                EARSum += EAR
                EARAddedTime += 1
    else: #caliEnd is True:
        if EAR is not -1 and headArea is not -1:
            if headArea < headAvg * 0.75:
                headStatus = 2
            else:
                headStatus = 3

            #0: forward
            #1: horizontal movement
            #2: vertical(upper) movement
            if headPose[0] < headPoseAvg[0]-8.5 or headPose[0] > headPoseAvg[0]+8.5:
                headPoseStatus = 1
            elif headPose[1] > headPoseAvg[1]+8.5:
                headPoseStatus = 2
            else:
                headPoseStatus = 0

            earCurTime = time.time()
            if EAR < EARAvg * 0.875:
                if EARTime is None:
                    EARTime = earCurTime
                else:
                    if earCurTime - EARTime > 3:#TIMETHRESHOLD
                        EARStatus = True
            else:
                #EARNOTDETECTEDTHISTIME
                EARStatus = False
                EARTime = None
        else:
            headStatus = 1
            EARStatus = False #Controversial!!!
    ### FOR YOLO ###
    
    local_prediction = int(daisee.prediction(frame))
        
    if headStatus is 1 or headStatus is 2 or headPoseStatus is 1 or headPoseStatus is 2:
        local_prediction = 0
    elif EARStatus is True:
        local_prediction = 0

    predEngage = dQ.detectionPush(local_prediction)
    if predEngage is 0:
        colorStatus = 0
    elif predEngage is 1:
        colorStatus = 1
    else:
        colorStatus = 2

    generalStatus = [False, False, False, False]
    if headStatus is None:
        pass
    elif headStatus is 1:
        generalStatus[0] = True
        generalStatus[1] = False
    elif headStatus is 2:
        generalStatus[0] = False
        generalStatus[1] = True    
    else:
        generalStatus[0] = False
        generalStatus[1] = False

    if headStatus is not 1 and generalStatus[1] is False: #When face is detected
        if headPoseStatus is None:
            pass
        elif headPoseStatus is 1:
            generalStatus[1] = True
        elif headPoseStatus is 2:
            generalStatus[1] = True
        else:
            generalStatus[1] = False

    if EARStatus is True:
        generalStatus[3] = True
    else:
        generalStatus[3] = False

    returnVal = [0, 0, 0, 0]
    if generalStatus[0] is True:
        returnVal[0] = 1
    if generalStatus[1] is True:
        returnVal[1] = 1
    if generalStatus[2] is True:
        returnVal[2] = 1
    if generalStatus[3] is True:
        returnVal[3] = 1

    outputJson = json.dumps({
        'colorStat': str(colorStatus),
        'generalStat': f'{returnVal[0]}{returnVal[1]}{returnVal[2]}{returnVal[3]}',
        'handStat': f'{handGestureStatus}'
        })
    print(outputJson)
    return outputJson

