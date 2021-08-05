import threading

from matplotlib.pyplot import tricontour

import cv2
import numpy as np
import os
from math import cos, sin
from PIL import Image
import time
import microphone_checker_stream
#from HeadPoseRevised import process_detection
#from EAR import calculate_ear
# import microphone_checker_stream
# import VoiceActivityDetection
from waiting import wait
import json
from queue import Queue
import datetime
from playsound import playsound
import daiseecnn
import csv
import torch
import glob
import YOLODetection
from YOLO.yolo_postprocess import YOLO
from tensorflow.python.framework.ops import disable_eager_execution
from YOLODetection import process_detection
import single_live_vad
from multiprocessing import Process
from EAR import calculate_ear

from flask import Flask, render_template, Response, jsonify, send_file, request

app = Flask(__name__)

# torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')
# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#                               model='silero_vad',
#                               force_reload=True)

# (get_speech_ts,
#  get_speech_ts_adaptive,
#  save_audio,
#  read_audio,
#  state_generator,
#  single_audio_stream,
#  collect_chunks) = utils

# def sendMU():
#     return model, utils

@app.route('/')
def preform():
    print('///1///')
    return render_template('preform.html', send=1)


@app.route('/', methods=['POST'])
def index():
    """Video streaming home page."""
    print('///2///')
    
    disable_eager_execution()
    now = datetime.datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    templateData = {
        'title': 'Image Streaming',
        'time': timeString
    }

    global STD_HUMAN_LABEL
    global LABEL_VAL
    result = request.form
    STD_HUMAN_LABEL = int(result["calitime"])  # 사람별 or 라벨별 기준값
    LABEL_VAL = int(result["logtime"]) # 라벨값

    global systemEnded
    systemEnded = False

    global TIMETHRESHOLD
    TIMETHRESHOLD = 3

    global createdTag
    createdTag = [[], [], [], []]

    global isVideoEnded
    isVideoEnded = False

    global silence
    global size
    global logStartTime
    global logEndTime
    global logType
    global isVideoSystemReady
    global isLiveLocal
    global isRealDebug
    global returnCheck
    global isThreadStart
    global loadingComplete
    global g_frame
    global img1
    global img2
    global img3
    global img4
    global logStudentName

    global colorStatus #0: Red / 1: Cream / 2: Green
    colorStatus = [2, 2, 2, 2]
    
    global generalStatus # index 0: 자리비움 / 1: 면적 줄어듬 / 2: 발표안함
    generalStatus = [[False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False]]

    global predEngage
    predEngage = [-1, -1, -1, -1]
    
    global headStatus
    headStatus = [None, None, None, None]

    global EARStatus
    EARStatus = [False, False, False, False]

    global isStartAudio
    isStartAudio=False


    global logFile
    logFile = list(range(4))
    for i in range(0, 4):
        logFile[i] = open(f"evalLog{i + 1}.txt", "w")

    global labels
    labelFile = open('AllLabels.csv','r')
    labels = csv.reader(labelFile)

    global labelFile_0
    global labelFile_1
    global labelFile_2
    global labelFile_3
    labelFile_0 = open('./labelZero.txt')
    labelFile_1 = open('./labelOne.txt')
    labelFile_2 = open('./labelTwo.txt')
    labelFile_3 = open('./labelThree.txt')

    g_frame = None

    logStudentName = 0
    logStartTime = 0
    logEndTime = 0
    logType = 0
    silence = 1
    size = 1
    isVideoSystemReady = False
    isLiveLocal = 1  # Local
    isRealDebug = 0  # Debug
    returnCheck = 0
    loadingComplete = False

    global predOnce
    predOnce = False

    global isFrameReady
    isFrameReady = False

    
    gen_frame_thread = threading.Thread(target=real_gen_frames)
    gen_frame_thread.start()
    #webCam_thread = threading.Thread(target=justWebCAM)
    #webCam_thread.start()

    # audio_thread = threading.Thread(target=play_audio)
    # audio_thread.start()

    # threading.Thread(target=vad_ctrl).start()
    # Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio1.wav",)).start()
    # Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio2.wav",)).start()
    # Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio3.wav",)).start()
    # Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio4.wav",)).start()

    # threading.Thread(target=VoiceActivityDetection.vadStart, args=("SampleAudio1.wav",)).start()
    # threading.Thread(target=VoiceActivityDetection.vadStart, args=("SampleAudio2.wav",)).start()
    # threading.Thread(target=VoiceActivityDetection.vadStart, args=("SampleAudio3.wav",)).start()
    # threading.Thread(target=VoiceActivityDetection.vadStart, args=("SampleAudio4.wav",)).start()
    # threading.Thread(target=getVADdata).start()

    global g_webcamVC
    g_webcamVC = cv2.VideoCapture(0)
    g_webcamVC.set(3, 640)
    g_webcamVC.set(4, 480)
    wait(lambda: predOnce, timeout_seconds=120, waiting_for="Prediction Process id At Least")

    threading.Thread(target=single_live_vad.start_recording).start()

    return render_template('index.html', **templateData)

# def play_audio():
#     global isStartAudio
#     while True:
#         if isStartAudio is True:
#             print("hello")
#             playsound('SampleAudioAll.wav')
#             break
#         time.sleep(0.1)

def getLiveSC():
    while True:
        global liveSC
        liveSC=single_live_vad.returnLiveSC()
        time.sleep(0.5)
    
# @app.route('/mfccstart', methods=['POST'])
# def mfccstart():
#     # th = threading.Thread(target=mfcc_ctrl, args=())
#     # th.start()
#     # return render_template('temp.html', value=0)
#     print("Start")


def vad_ctrl():
   pass
    # SpeechCount1 = VoiceActivityDetection.getSC(1)
    # SpeechCount2 = VoiceActivityDetection.getSC(2)
    # SpeechCount3 = VoiceActivityDetection.getSC(3)
    # SpeechCount4 = VoiceActivityDetection.getSC(4)

    # print("\n\n\n디버기이잉", SpeechCount1, SpeechCount2, SpeechCount3, SpeechCount4)

def real_gen_frames():
    daisee = daiseecnn.DaiseeCNN()
    yolo = YOLO()

    global loadingComplete
    loadingComplete = True
    global isLiveLocal
    global g_frame

    peerNum = 4
    # 조건 검사를 위한 변수들
    frCnt = 0

    CALITIME = 13
    caliEnd = False

    startTime = time.time()
    prevTime = startTime
    global STD_HUMAN_LABEL
    global LABEL_VAL
    global TIMETHRESHOLD
    #print(str(LABEL_VAL - 0))
    CNNTime = [None, None, None, None]

    headSum = [0, 0, 0, 0]
    headAvg = [None, None, None, None]
    
    EARSum = [0, 0, 0, 0]
    EARAvg = [None, None, None, None]
    EARTime = [None, None, None, None]

    # log 남기기 위한 global vars
    global logStudentName
    global logStartTime
    global logEndTime
    global logType
    logStudentName = 0
    logStartTime = 0
    logEndTime = 0
    logType = 0

    # variables for frame control. never be used in live detection.
    frameTemp = 0
    frameCtrl = None
    frameBack = 0
    #dQ = [detectionQueue(), detectionQueue(), detectionQueue(), detectionQueue()]

    firstImg = cv2.imread('./FirstImage.jpg')
    global predOnce

    faceAddedTime = [0, 0, 0, 0]
    EARAddedTime = [0, 0, 0, 0]

    global EARStatus
    
    while True:
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

        if predOnce is False:
            print("predictMae")
            dummy = daisee.prediction(firstImg)
            predOnce = True
            startTime = time.time()
            continue

        if g_frame is None:
            continue
        
        frCnt += 1
        try:
            print('FPS: ' + str(1 / (sec)))
        except:
            print('FPS___')

        # if frameCtrl is None:
        #     frameBack = frCnt
        # else:
        #     frameBack = frameCtrl
        #     frCnt = frameCtrl
        #     print('move back')
        #     frameCtrl = None

        global returnCheck
        if returnCheck == 1:
            logStudentName = 0
            logStartTime = 0
            logEndTime = 0
            logType = 0
            returnCheck = 0

        #isDetectedOnce = [0, 0, 0, 0]

        sendedFrame = list(range(peerNum))
        for i in range(0, peerNum):
            sendedFrame[i] = g_frame[i]
        try:
            rgbFrame = list(range(peerNum))
            for i in range(0, peerNum):
                rgbFrame[i] = cv2.cvtColor(sendedFrame[i], cv2.COLOR_BGR2RGB)
        except:
            #print("Couldn't grapped. Maybe video is changing..")
            continue
            #break

        ### FOR YOLO ###
        
        global headStatus
        # !설명 headStatus:
        # None: Calibationing
        # 1: cannot recognize face
        # 2: face detected is too smaller than avg
        # 3: normal!

        img_pil = list(range(peerNum))
        for i in range(0, peerNum):
            img_pil[i] = Image.fromarray(rgbFrame[i])

        bboxes = list(range(peerNum))
        scores = list(range(peerNum))
        classes = list(range(peerNum))
        for i in range(0, peerNum):
            bboxes[i], scores[i], classes[i] = yolo.YOLO_DetectProcess(img_pil[i])

        headArea = list(range(peerNum))
        faceDetected = list(range(peerNum))
        for i in range(0, peerNum):
            faceDetected[i] = True
            if len(bboxes[i]) is 0:
                faceDetected[i] = False
            else:
                headArea[i] = process_detection(sendedFrame[i], bboxes[i][0])# reason of referencing index 0 in bboxes:
                #print(f'area:{headArea[i]}')
                #print(f'{i+1}: {headArea[i]}')
        print()

        EAR = list(range(peerNum))
        for i in range(0, peerNum):
            EAR[i] = calculate_ear(rgbFrame[i])

        if caliEnd is False:
            if time.time() - startTime > CALITIME: # CALIBRATION DONE
                #print('/////////////////////////')
                #print('////////CALIDONE/////////')
                #print('/////////////////////////')
                
                for i in range(0, peerNum):
                    headAvg[i] = headSum[i] / faceAddedTime[i]
                    EARAvg[i] = EARSum[i] / EARAddedTime[i]
                    #print(f'AVG: {headAvg[i]}')
                caliEnd = True
            else:
                for i in range(0, peerNum):
                    if faceDetected[i] is True:
                        headSum[i] += headArea[i]
                        faceAddedTime[i] += 1
                    if EAR[i] is not None:
                        EARSum[i] += EAR[i]
                        EARAddedTime[i] += 1
        else: #caliEnd is True:
            for i in range(0, peerNum):
                if faceDetected[i] is False:
                    headStatus[i] = 1
                else:
                    if headArea[i] < headAvg[i] * 0.8:
                        headStatus[i] = 2
                    else:
                        headStatus[i] = 3

                if EAR[i] is not None:
                    earCurTime = time.time()
                    if EAR[i] < EARAvg[i] * 0.91:
                        if EARTime[i] is None:
                            EARTime[i] = earCurTime
                        else:
                            if earCurTime - EARTime[i] > 3:#TIMETHRESHOLD
                                EARStatus[i] = True
                    else:
                        #EARNOTDETECTEDTHISTIME
                        EARStatus[i] = False
                        EARTime[i] = None
        ### FOR YOLO ###

        
        global predEngage
        local_prediction = list(range(peerNum))
        for i in range(0, peerNum):
            local_prediction[i] = int(daisee.prediction(sendedFrame[i]))
            
        for i in range(0, peerNum):
            if headStatus[i] is 1 or headStatus[i] is 2:
                predEngage[i] = 0
            elif EARStatus[i] is True:
                predEngage[i] = 0
            else:
                predEngage[i] = local_prediction[i]

        

        # for i in range(0, peerNum):
        #     if predEngage[i] is 0 or predEngage[i] is 1:
        #         if CNNTime[i] is None:
        #             CNNTime[i] = time.time()
        #             frameTemp = frCnt
        #         else:
        #             cnnCurTime = time.time()
        #             if cnnCurTime - CNNTime[i] > TIMETHRESHOLD:
        #                 pass
        #     else:
        #         if CNNTime[i] is not None and time.time() - CNNTime[i] > TIMETHRESHOLD:
        #             pass #LOG남기는Action
        #         CNNTime[i] = None

    #never reached 08/05
    global systemEnded
    systemEnded = True


def justWebCAM():
    myCap = cv2.VideoCapture(0)
    myCap.set(3, 1280)  # CV_CAP_PROP_FRAME_WIDTH
    myCap.set(4, 720)  # CV_CAP_PROP_FRAME_HEIGHT
    
    myRet, myFrame = myCap.read()
    global predOnce
    wait(lambda: predOnce, timeout_seconds=120, waiting_for="video process ready")

    while True:
        myRet, myFrame = myCap.read()

        if myRet:
            ret, buffer = cv2.imencode('.jpg', myFrame)
            myBytes = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + myBytes + b'\r\n')

class detectionQueue:
    def __init__(self):
        self.que = Queue()
        self.size = 0
        self.sum = 0
        self.avg = 0

        self.yelloThresh = 4
        self.redThresh = 6
        self.maxSize = 10

    def detectionPush(self, data):
        if self.size < self.maxSize:
            self.que.put(data)
            self.size += 1
            self.sum += data
            self.avg = self.sum / self.size

            return 0
        else:
            deq = self.que.get()
            self.que.put(data)
            self.sum += data
            self.sum -= deq
            self.avg = self.sum / self.size

            if self.avg > (self.redThresh / self.maxSize):
                return 3
            elif self.avg > (self.yelloThresh / self.maxSize):
                return 2
            else:
                return 1

class Streaming:
    def __init__(self, peer):
        peer = int(peer)
        global isLiveLocal
        global frameReady
        # global loadingComplete
        global g_frame
        global STD_HUMAN_LABEL
        self.peerNum = 4

        g_frame = list(range(self.peerNum))
        frameReady = list(range(self.peerNum))

        self.redImg = np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8)
        self.greenImg = np.full((480, 640, 3), (0, 255, 0), dtype=np.uint8)
        self.neutralImg = np.full((480, 640, 3), (160, 172, 203), dtype=np.uint8)
        self.blueImg = np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8)
        self.yelloImg = np.full((480, 640, 3), (0, 255, 255), dtype=np.uint8)

        if isLiveLocal is 0:
            self.srcPath = 0
        elif peer is 1:
            global g_webcamVC
            self.srcPath = 0
            self.cap = g_webcamVC
        else:
            if STD_HUMAN_LABEL is 1:
                self.srcFold = f'./selfTestVideos/{peer}/'
                self.srcVideoText = open(f'./selfTestVideos/{peer}/videos.txt')
                self.humanLabelText = open(f'./selfTestVideos/{peer}/humanMadeLabel.txt')
            else:
                self.srcFold = f'./selfTestVideos/daisee/{peer}/'
                self.srcVideoText = open(f'./selfTestVideos/daisee/{peer}/videos.txt')
                self.humanLabelText = open(f'./selfTestVideos/daisee/{peer}/humanMadeLabel.txt')

            self.srcPath = self.srcFold + self.srcVideoText.readline()
            self.humanLabel = self.humanLabelText.readline()
            self.cap = cv2.VideoCapture(self.srcPath)

        # wait(lambda: loadingComplete, timeout_seconds=120, waiting_for="video process ready")
        # self.labeledEngagement = -1

        # global labels
        # for line in labels:
        #     if line[0] == self.srcPath :
        #         self.labeledEngagement = int(line[2])
        #         break

        global mfccStartTime
        mfccStartTime = time.time()

    def local_frames(self, peer):
        global g_frame
        global isVideoSystemReady
        global isLiveLocal
        global isRealDebug
        global frameReady
        global predOnce
        global colorStatus
        global generalStatus

        peer = int(peer)
        ret, g_frame[peer - 1] = self.cap.read()
        
        wait(lambda: predOnce, timeout_seconds=120, waiting_for="Prediction Process id At Least")
        global isStartAudio
        if self.cap.isOpened():
            while True:
                ret, g_frame[peer - 1] = self.cap.read()
                if ret:
                    global predEngage
                    global headStatus
                    global EARStatus

                    #print(f'peer:{peer - 1} and pred: {predEngage[peer - 1]}')
                    if predEngage[peer - 1] is -1: #undefined
                        l_frame = g_frame[peer - 1]
                    elif predEngage[peer - 1] is 0 or predEngage[peer - 1] is 1: #not engaged
                        l_frame = cv2.addWeighted(self.redImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (640, 480), (0, 0, 255), 20)
                        # cv2.putText(l_frame, f'Predict: {predEngage[peer - 1]}', (10, 100),
                        #         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        #         cv2.LINE_AA)
                        colorStatus[peer - 1] = 0
                    elif predEngage[peer - 1] is 2: #neutral
                        l_frame = cv2.addWeighted(self.neutralImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (640, 480), (160, 172, 203), 20)
                        # cv2.putText(l_frame, f'Predict: {predEngage[peer - 1]}', (10, 100),
                        #         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        #         cv2.LINE_AA)
                        colorStatus[peer - 1] = 1
                    else: #highly engaged
                        l_frame = cv2.addWeighted(self.greenImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (640, 480), (0, 255, 0), 20)
                        # cv2.putText(l_frame, f'Predict: {predEngage[peer - 1]}', (10, 100),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        #     cv2.LINE_AA)
                        colorStatus[peer - 1] = 2

                    if headStatus[peer - 1] is None:
                        # cv2.putText(l_frame, f'HeadDt: Calibrationing...Do wait.', (10, 400),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        #     cv2.LINE_AA)
                        pass
                    elif headStatus[peer - 1] is 1:
                        # cv2.putText(l_frame, f'HeadDt: Cannot Have Detected Face!', (10, 400),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        #     cv2.LINE_AA)
                        generalStatus[peer - 1][0] = True
                        generalStatus[peer - 1][1] = False
                    elif headStatus[peer - 1] is 2:
                        # cv2.putText(l_frame, f'HeadDt: Detected Decreased Face', (10, 400),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        #     cv2.LINE_AA)
                        generalStatus[peer - 1][0] = False
                        generalStatus[peer - 1][1] = True
                    else:
                        # cv2.putText(l_frame, f'HeadDt: NORMAL!', (10, 400),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        #     cv2.LINE_AA)
                        generalStatus[peer - 1][0] = False
                        generalStatus[peer - 1][1] = False

                    if EARStatus[peer - 1] is True:
                        generalStatus[peer - 1][3] = True
                    else:
                        generalStatus[peer - 1][3] = False
                    
                    if peer is not 1:
                        if int(self.humanLabel) is 0:
                            cv2.rectangle(l_frame, (11, 11), (30, 30), (0, 0, 255), -1)
                        elif int(self.humanLabel) is 1:
                            cv2.rectangle(l_frame, (11, 11), (30, 30), (160, 172, 203), -1)
                        else:
                            cv2.rectangle(l_frame, (11, 11), (30, 30), (0, 255, 0), -1)

                    ret, buffer = cv2.imencode('.jpg', l_frame)
                    l_frame = buffer.tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + l_frame + b'\r\n')
                    isStartAudio=True
                else:
                    if peer is not 1:
                        self.srcPath = self.srcVideoText.readline()
                        self.humanLabel = self.humanLabelText.readline()
                        
                        print('Done?')
                        if not self.srcPath:
                            colorStatus = [-1, -1, -1, -1]
                            global isVideoEnded
                            isVideoEnded = False
                            
                            wait(lambda: isVideoEnded, timeout_seconds=120, waiting_for="video rewind process")

                            #REWIND
                            self.srcVideoText.seek(0)
                            self.humanLabelText.seek(0)
                            colorStatus = [2, 2, 2, 2]

                            continue
                        self.srcPath = self.srcFold + self.srcPath
                        self.cap = cv2.VideoCapture(self.srcPath)
                    else:
                        print('webcam error.')

                # if isLiveLocal is 1:
                #     if cv2.waitKey(15) & 0xFF == ord('q'):  # press q to quit
                #         break
                if STD_HUMAN_LABEL is 1:
                    if isLiveLocal is 1:
                        if cv2.waitKey(15) & 0xFF == ord('q'):  # press q to quit
                            break
                else:
                    if isLiveLocal is 1:
                        if cv2.waitKey(7) & 0xFF == ord('q'):  # press q to quit
                            break

        else:
            print('Cannot Open File ERR')

class Streaming_LabelBased:
    def __init__(self, peer, level):
        peer = int(peer)
        global isLiveLocal
        global frameReady
        # global loadingComplete
        global g_frame
        global labelFile_0
        global labelFile_1
        global labelFile_2
        global labelFile_3

        self.peerNum = 4
        self.engageLevel = level

        g_frame = list(range(self.peerNum))
        frameReady = list(range(self.peerNum))

        self.redImg = np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8)
        self.greenImg = np.full((480, 640, 3), (0, 255, 0), dtype=np.uint8)
        self.neutralImg = np.full((480, 640, 3), (160, 172, 203), dtype=np.uint8)
        self.blueImg = np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8)
        self.yelloImg = np.full((480, 640, 3), (0, 255, 255), dtype=np.uint8)

        if isLiveLocal is 0:
            self.srcPath = 0
        elif peer is 1:
            global g_webcamVC
            self.srcPath = 0
            self.cap = g_webcamVC
        else:
            if self.engageLevel is 0:
                self.srcPath = labelFile_0.readline()
            elif self.engageLevel is 1:
                self.srcPath = labelFile_1.readline()
            elif self.engageLevel is 2:
                self.srcPath = labelFile_2.readline()
            elif self.engageLevel is 3:
                self.srcPath = labelFile_3.readline()
            self.srcPath = './DAiSEE/DataSet/Test/' + self.srcPath[0:6] + '/' + self.srcPath[0:-4] + '/' + self.srcPath

            self.cap = cv2.VideoCapture(self.srcPath)

        # wait(lambda: loadingComplete, timeout_seconds=120, waiting_for="video process ready")
        # self.labeledEngagement = -1

        # global labels
        # for line in labels:
        #     if line[0] == self.srcPath:
        #         self.labeledEngagement = int(line[2])
        #         break

        global mfccStartTime
        mfccStartTime = time.time()

    def local_frames(self, peer):
        global g_frame
        global isVideoSystemReady
        global isLiveLocal
        global isRealDebug
        global frameReady
        global LABEL_VAL
        global predOnce

        global colorStatus
        global generalStatus

        peer = int(peer)
        ret, g_frame[peer - 1] = self.cap.read()

        wait(lambda: predOnce, timeout_seconds=120, waiting_for="Prediction Process id At Least")
        global isStartAudio
        if self.cap.isOpened():
            while True:
                ret, g_frame[peer - 1] = self.cap.read()
                if ret:
                    global predEngage
                    global headStatus
                    global EARStatus

                    #print(f'peer:{peer - 1} and pred: {predEngage[peer - 1]}')
                    if predEngage[peer - 1] is -1: #undefined
                        l_frame = g_frame[peer - 1]
                    elif predEngage[peer - 1] is 0 or predEngage[peer - 1] is 1: #not engaged
                        l_frame = cv2.addWeighted(self.redImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (640, 480), (0, 0, 255), 20)
                        # cv2.putText(l_frame, f'Engagement: {LABEL_VAL}', (10, 50),
                        #         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2,
                        #         cv2.LINE_AA)
                        # cv2.putText(l_frame, f'Prediction: {predEngage[peer - 1]}', (10, 100),
                        #         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        #         cv2.LINE_AA)
                        colorStatus[peer - 1] = 0
                    elif predEngage[peer - 1] is 2: #neutral
                        l_frame = cv2.addWeighted(self.neutralImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (640, 480), (160, 172, 203), 20)
                        # cv2.putText(l_frame, f'Engagement: {LABEL_VAL}', (10, 50),
                        #         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2,
                        #         cv2.LINE_AA)
                        # cv2.putText(l_frame, f'Prediction: {predEngage[peer - 1]}', (10, 100),
                        #         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        #         cv2.LINE_AA)
                        colorStatus[peer - 1] = 1
                    else: #highly engaged
                        l_frame = cv2.addWeighted(self.greenImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (640, 480), (0, 255, 0), 20)
                        # cv2.putText(l_frame, f'Engagement: {LABEL_VAL}', (10, 50),
                        #         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2,
                        #         cv2.LINE_AA)
                        # cv2.putText(l_frame, f'Prediction: {predEngage[peer - 1]}', (10, 100),
                        #         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        #         cv2.LINE_AA)
                        colorStatus[peer - 1] = 2

                    if headStatus[peer - 1] is None:
                        # cv2.putText(l_frame, f'HeadDt: Calibrationing...Do wait.', (10, 400),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
                        pass
                    elif headStatus[peer - 1] is 1:
                        # cv2.putText(l_frame, f'HeadDt: Cannot Have Detected Face!', (10, 400),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
                        generalStatus[peer - 1][0] = True
                        generalStatus[peer - 1][1] = False
                    elif headStatus[peer - 1] is 2:
                        # cv2.putText(l_frame, f'HeadDt: Detected Decreased Face', (10, 400),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
                        generalStatus[peer - 1][0] = False
                        generalStatus[peer - 1][1] = True    
                    else:
                        # cv2.putText(l_frame, f'HeadDt: NORMAL!', (10, 400),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
                        generalStatus[peer - 1][0] = False
                        generalStatus[peer - 1][1] = False
                    
                    if EARStatus[peer - 1] is True:
                        generalStatus[peer - 1][3] = True
                    else:
                        generalStatus[peer - 1][3] = False

                    if peer is not 1:
                        if int(LABEL_VAL) is 0 or int(LABEL_VAL) is 1:
                            cv2.rectangle(l_frame, (11, 11), (30, 30), (0, 0, 255), -1)
                        elif int(LABEL_VAL) is 2:
                            cv2.rectangle(l_frame, (11, 11), (30, 30), (160, 172, 203), -1)
                        else:
                            cv2.rectangle(l_frame, (11, 11), (30, 30), (0, 255, 0), -1)
                    
                    ret, buffer = cv2.imencode('.jpg', l_frame)
                    l_frame = buffer.tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + l_frame + b'\r\n')
                    isStartAudio=True
                else:
                    if peer is not 1:
                        if self.engageLevel is 0:
                            self.srcPath = labelFile_0.readline()
                        elif self.engageLevel is 1:
                            self.srcPath = labelFile_1.readline()
                        elif self.engageLevel is 2:
                            self.srcPath = labelFile_2.readline()
                        elif self.engageLevel is 3:
                            self.srcPath = labelFile_3.readline()

                        if not self.srcPath:
                            colorStatus = [-1, -1, -1, -1]

                            global isVideoEnded
                            isVideoEnded = False

                            wait(lambda: loadingComplete, timeout_seconds=120, waiting_for="video process ready")
                            #REWIND
                            self.srcVideoText.seek(0)
                            colorStatus = [2, 2, 2, 2]

                            continue

                        self.srcPath = './DAiSEE/DataSet/Test/' + self.srcPath[0:6] + '/' + self.srcPath[
                                                                                            0:-4] + '/' + self.srcPath
                        self.cap = cv2.VideoCapture(self.srcPath)
                    else:
                        print('WebCam Reading Error. Skipping this frame')

                if isLiveLocal is 1:
                    if cv2.waitKey(10) & 0xFF == ord('q'):  # press q to quit
                        break

        else:
            print('Cannot Open File ERR')


@app.route('/video_feed/<peer>')
def video_feed(peer):
    """Video streaming route. Put this in the src attribute of an img tag."""
    global isRealDebug
    global frameReady
    global loadingComplete
    global STD_HUMAN_LABEL
    global LABEL_VAL
    #wait(lambda: loadingComplete, timeout_seconds=120, waiting_for="video process ready")
    #cv2.waitKey(200)
    if isRealDebug is 0:
        return Response(Streaming(peer).local_frames(peer),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        pass

        # if frameReady[0] is True:
        #     return Response(frameSession1(),
        #                 mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/justWebCAM_feed')
def justWebCAM_feed():
    return Response(justWebCAM(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_rewind')
def video_rewind():
    global isVideoEnded
    global colorStatus #0: Red / 1: Cream / 2: Green
    isVideoEnded = True
    colorStatus = [2, 2, 2, 2]

    return (''), 204


@app.route('/mfccend', methods=['POST'])
def mfccend():
    # microphone_checker_stream.process_stop()
    # mfcc_ctrl()
    return render_template('temp.html', value=0)

@app.route('/colorStat_feed', methods=['POST'])
def colorStat_feed():
    global colorStatus

    return jsonify({
        'stu1Color': str(colorStatus[0]),
        'stu2Color': str(colorStatus[1]),
        'stu3Color': str(colorStatus[2]),
        'stu4Color': str(colorStatus[3])
    })

@app.route('/generalStat_feed', methods=['POST'])
def generalStat_feed():
    global generalStatus
    peerNum = 4
    returnVal = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    for i in range(0, peerNum):
        if generalStatus[i][0] is True:
            returnVal[i][0] = 1
        if generalStatus[i][1] is True:
            returnVal[i][1] = 1
        if generalStatus[i][2] is True:
            returnVal[i][2] = 1
        if generalStatus[i][3] is True:
            returnVal[i][3] = 1

    sc1 = single_live_vad.returnLiveSC()
    sc2=sc3=sc4=0
    scAVG = (sc1+sc2+sc3+sc4)/4
    if(sc1<scAVG):
        returnVal[0][2]=1
    if(sc2<scAVG):
        returnVal[1][2]=1
    if(sc3<scAVG):
        returnVal[2][2]=1
    if(sc4<scAVG):
        returnVal[3][2]=1
   
    return jsonify({
        'stu1General': f'{returnVal[0][0]}{returnVal[0][1]}{returnVal[0][2]}{returnVal[0][3]}',
        'stu2General': f'{returnVal[1][0]}{returnVal[1][1]}{returnVal[1][2]}{returnVal[1][3]}',
        'stu3General': f'{returnVal[2][0]}{returnVal[2][1]}{returnVal[2][2]}{returnVal[2][3]}',
        'stu4General': f'{returnVal[3][0]}{returnVal[3][1]}{returnVal[3][2]}{returnVal[3][3]}'
    })


@app.route('/vad_feed', methods=['POST'])
def vad_feed():
    # global silence
    # global size
    # sc1 = VoiceActivityDetection.setSC1()
    # sc2 = VoiceActivityDetection.setSC2()
    # sc3 = VoiceActivityDetection.setSC3()
    # sc4 = VoiceActivityDetection.setSC4()
    # sc1 = 0
    global sc1, sc2, sc3, sc4
    sc1 = single_live_vad.returnLiveSC()

    #print("\n\n이거가 라이브 횟수", sc1)
    sc2 = 0
    sc3 = 0
    sc4 = 0
    return jsonify({
        'SpeechCount1': str(sc1),
        'SpeechCount2': str(sc2),
        'SpeechCount3': str(sc3),
        'SpeechCount4': str(sc4)
    })

# @app.route('/fig1', methods=['POST'])
# def fig1():
#     global img1
#     #img1 = VoiceActivityDetection.getPlot(1)
#     return send_file(img1, mimetype='image/png')

# @app.route('/fig2', methods=['POST'])
# def fig2():
#     global img2
#     #img2 = VoiceActivityDetection.getPlot(2)
#     return send_file(img2, mimetype='image/png')

# @app.route('/fig3')
# def fig3():
#     global img3
#     #img3 = VoiceActivityDetection.getPlot(3)
#     return send_file(img3, mimetype='image/png')

# @app.route('/fig4')
# def fig4():
#     global img4
#     #img4 = VoiceActivityDetection.getPlot(4)
#     return send_file(img4, mimetype='image/png')


@app.route('/log_feed', methods=['POST'])
def log_feed():
    global logStartTime
    global logEndTime
    global logType
    global returnCheck
    global logStudentName
    global logFile
    global createdTag

    if logStartTime is not 0 and logEndTime is not 0 and logType is not 0:
        #print(f"fStinghatton: {round(logStartTime, 1)} {round(logEndTime, 1)} {logType}\n")

        #NO LOG IN THIS VERSION
        # logFile[logStudentName].write(f"{round(logStartTime, 1)} {round(logEndTime, 1)} {logType}\n")
        # createdTag[logStudentName].append([round(logStartTime), round(logEndTime), logType])
        returnCheck = 1

    return jsonify({
        'name': str(logStudentName),
        'startTime': str(logStartTime),
        'endTime': str(logEndTime),
        'behaviorType': str(logType)
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1')