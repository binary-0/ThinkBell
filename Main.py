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
import VoiceActivityDetection
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
import globalVAR
import matplotlib.pylab as plt
from io import BytesIO


from multiprocessing import Process, Queue

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

global plotImg

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

    global CALITIME
    global TIMETHRESHOLD
    result = request.form
    CALITIME = int(result["calitime"])
    TIMETHRESHOLD = int(result["logtime"])
    global systemEnded
    systemEnded = False

    global createdTag
    createdTag = [[], [], [], []]

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
    colorStatus = [2, 2, 2, 2, 2]
    global generalStatus # index 0: 자리비움 / 1: 면적 줄어듬 / 2: 발표안함
    generalStatus = [[False, False, False], [False, False, False], [False, False, False], [False, False, False], [False, False, False]]

    global predEngage
    predEngage = [-1, -1, -1, -1, -1]

    global isStartAudio
    isStartAudio=False

    # global globalVAR.vc1, globalVAR.vc2, globalVAR.vc3, globalVAR.vc4
    
    
    global logFile
    logFile = list(range(4))
    for i in range(0, 4):
        logFile[i] = open(f"evalLog{i + 1}.txt", "w")

    global labels    
    labelFile = open('AllLabels.csv','r')
    labels = csv.reader(labelFile)

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

    global headStatus
    headStatus = [None, None, None, None, None]
    
    gen_frame_thread = threading.Thread(target=real_gen_frames)
    gen_frame_thread.start()
    #webCam_thread = threading.Thread(target=justWebCAM)
    #webCam_thread.start()

    audio_thread = threading.Thread(target=play_audio)
    audio_thread.start()

    # threading.Thread(target=vad_ctrl).start()
    # Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio1.wav",)).start()
    # Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio2.wav",)).start()
    # Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio3.wav",)).start()
    # Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio4.wav",)).start()

    global g_webcamVC
    g_webcamVC = cv2.VideoCapture(0)
    g_webcamVC.set(3, 640)
    g_webcamVC.set(4, 480)
    wait(lambda: predOnce, timeout_seconds=120, waiting_for="Prediction Process id At Least")

    threading.Thread(target=VoiceActivityDetection.vadStart, args=("SampleAudio1.wav",)).start()
    threading.Thread(target=VoiceActivityDetection.vadStart, args=("SampleAudio2.wav",)).start()
    threading.Thread(target=VoiceActivityDetection.vadStart, args=("SampleAudio3.wav",)).start()
    threading.Thread(target=VoiceActivityDetection.vadStart, args=("SampleAudio4.wav",)).start()  
    threading.Thread(target=getVADdata).start()

    # while True:
    #     print(vc1, vc2, vc3, vc4)
    #     time.sleep(500)
    return render_template('index.html', **templateData)

plt.rcParams["figure.figsize"]=(12,8)

def getVADdata():
    while True: 
        global vc1, vc2, vc3, vc4
        global sc1, sc2, sc3, sc4
        vc1,sc1 = VoiceActivityDetection.setVADdata(1)
        vc2,sc2 = VoiceActivityDetection.setVADdata(2)
        vc3,sc3 = VoiceActivityDetection.setVADdata(3)
        vc4,sc4 = VoiceActivityDetection.setVADdata(4)
        
        plt.ylim([0,1])
        plt.xticks([])
        plt.axhline(y=0.7, color='r')
        fig, axs = plt.subplots(4)

        axs[0].plot(vc1)
        axs[0].set_ylim([0,0.7])
        axs[0].axhline(y=0.5, color='r')
        axs[1].plot(vc2)
        axs[1].set_ylim([0,0.7])
        axs[1].axhline(y=0.5, color='r')
        axs[2].plot(vc3)
        axs[2].set_ylim([0,0.7])
        axs[2].axhline(y=0.5, color='r')
        axs[3].plot(vc4)
        axs[3].set_ylim([0,0.7])
        axs[3].axhline(y=0.5, color='r')
        imgBytes = BytesIO()
        plt.savefig(imgBytes, format='png', bbox_inches='tight', dpi=200)
        yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + imgBytes.getvalue() + b'\r\n')
        # global plotImg
        # plotImg = imgBytes
        # temp1 = VoiceActivityDetection.setVC1()
        # temp2 = VoiceActivityDetection.setVC2()
        # print(temp1)
        # print("\n\n\n", temp2, "시발\n\n\n\n")
        # time.sleep(1)
        # plt.plot(vc1)
        # plt.pause(0.000001)
        # plt.clf()
        # plt.clf()
        # time.sleep(1)
        # print(vc1[:5],vc2[:5],vc3[:5],vc4[:5],"\n\n\n\n")
        


def play_audio():
    global isStartAudio
    while True:
        if isStartAudio is True:
            print("hello")
            playsound('SampleAudioAll.wav')
            break
        time.sleep(0.1)


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
    peerNum = 5
    # 조건 검사를 위한 변수들
    frCnt = 0

    CALITIME = 30
    caliEnd = False

    startTime = time.time()
    prevTime = startTime
    global CALITIME
    global TIMETHRESHOLD
    print(str(LABEL_VAL - 0))
    CNNTime = [None, None, None, None, None]

    headSum = [0, 0, 0, 0, 0]
    headAvg = [None, None, None, None, None]

    # log 남기기 위한 global vars
    global logStudentName
    global logStartTime
    global logEndTime
    global logType
    logStudentName = 0
    logStartTime = 0
    logEndTime = 0
    logType = 0

    frameTemp = 0
    frameCtrl = None
    frameBack = 0
    #dQ = [detectionQueue(), detectionQueue(), detectionQueue(), detectionQueue()]

    time.sleep(3)

    faceAddedTime = [0, 0, 0, 0, 0]
    
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
            print("Video Ended.")
            break

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
                print(f'area:{headArea[i]}')
                #print(f'{i+1}: {headArea[i]}')
        print()

        if caliEnd is False:
            if time.time() - startTime > CALITIME: # CALIBRATION DONE
                print('/////////////////////////')
                print('////////CALIDONE/////////')
                print('/////////////////////////')
                
                for i in range(0, peerNum):
                    headAvg[i] = headSum[i] / faceAddedTime[i]
                    print(f'AVG: {headAvg[i]}')
                caliEnd = True
            else:
                for i in range(0, peerNum):
                    if faceDetected[i] is True:
                        headSum[i] += headArea[i]
                        faceAddedTime[i] += 1
        else: #caliEnd is True:
            for i in range(0, peerNum):
                if faceDetected[i] is False:
                    headStatus[i] = 1
                else:
                    if headArea[i] < headAvg[i] * 0.85:
                        headStatus[i] = 2
                    else:
                        headStatus[i] = 3
        ### FOR YOLO ###
        
        global predEngage
        for i in range(0, peerNum):
            predEngage[i] = int(daisee.prediction(sendedFrame[i]))

        for i in range(0, peerNum):
            if predEngage[i] is 0 or predEngage[i] is 1:
                if CNNTime[i] is None:
                    CNNTime[i] = time.time()
                    frameTemp = frCnt
                else:
                    cnnCurTime = time.time()
                    if cnnCurTime - CNNTime[i] > TIMETHRESHOLD:
                        pass
            else:
                if CNNTime[i] is not None and time.time() - CNNTime[i] > TIMETHRESHOLD:
                    pass #LOG남기는Action
                CNNTime[i] = None

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


class Streaming:
    def __init__(self, peer):
        peer = int(peer)
        global isLiveLocal
        global frameReady
        # global loadingComplete
        global g_frame
        self.peerNum = 5

        g_frame = list(range(self.peerNum))
        frameReady = list(range(self.peerNum))

        self.redImg = np.full((720, 1280, 3), (0, 0, 255), dtype=np.uint8)
        self.greenImg = np.full((720, 1280, 3), (0, 255, 0), dtype=np.uint8)
        self.blueImg = np.full((720, 1280, 3), (255, 0, 0), dtype=np.uint8)
        self.yelloImg = np.full((720, 1280, 3), (0, 255, 255), dtype=np.uint8)

        if isLiveLocal is 0:
            self.srcPath = 0
        elif peer is 5:
            global g_webcamVC
            self.srcPath = 0
            self.cap = g_webcamVC
        else:
            self.srcPath = f'./SamV{peer}.mp4'
            self.cap = cv2.VideoCapture(self.srcPath)

        # wait(lambda: loadingComplete, timeout_seconds=120, waiting_for="video process ready")
        self.labeledEngagement = -1

        global labels
        for line in labels:
            if line[0] == self.srcPath :
                self.labeledEngagement = int(line[2])
                break

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

                    #print(f'peer:{peer - 1} and pred: {predEngage[peer - 1]}')
                    if predEngage[peer - 1] is -1: #undefined
                        l_frame = g_frame[peer - 1]
                    elif predEngage[peer - 1] is 0 or predEngage[peer - 1] is 1: #not engaged
                        l_frame = cv2.addWeighted(self.redImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (640, 480), (0, 0, 255), 20)
                        cv2.putText(l_frame, f'Predict: {predEngage[peer - 1]}', (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                        colorStatus[peer - 1] = 0                      
                    elif predEngage[peer - 1] is 2: #neutral
                        l_frame = cv2.addWeighted(self.neutralImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (640, 480), (160, 172, 203), 20)
                        cv2.putText(l_frame, f'Predict: {predEngage[peer - 1]}', (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                        colorStatus[peer - 1] = 1
                    else: #highly engaged
                        l_frame = cv2.addWeighted(self.greenImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (640, 480), (0, 255, 0), 20)
                        cv2.putText(l_frame, f'Predict: {predEngage[peer - 1]}', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                        colorStatus[peer - 1] = 2

                    if headStatus[peer - 1] is None:
                        cv2.putText(l_frame, f'HeadDt: Calibrationing...Do wait.', (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                    elif headStatus[peer - 1] is 1:
                        cv2.putText(l_frame, f'HeadDt: Cannot Have Detected Face!', (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                        generalStatus[peer - 1][0] = True
                        generalStatus[peer - 1][1] = False
                    elif headStatus[peer - 1] is 2:
                        cv2.putText(l_frame, f'HeadDt: Detected Decreased Face', (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                        generalStatus[peer - 1][0] = False
                        generalStatus[peer - 1][1] = True
                    else:
                        cv2.putText(l_frame, f'HeadDt: NORMAL!', (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                        generalStatus[peer - 1][0] = False
                        generalStatus[peer - 1][1] = False

                    ret, buffer = cv2.imencode('.jpg', l_frame)
                    l_frame = buffer.tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + l_frame + b'\r\n')
                    isStartAudio=True
                else:
                    print('Everything has done successfully.')
                    exit()
                    break

                if isLiveLocal is 1:
                    if cv2.waitKey(10) & 0xFF == ord('q'):  # press q to quit
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

        self.peerNum = 5
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
        elif peer is 5:
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

        peer = int(peer)
        ret, g_frame[peer - 1] = self.cap.read()

        wait(lambda: predOnce, timeout_seconds=120, waiting_for="Prediction Process id At Least")
        global isStartAudio
        if self.cap.isOpened():
            while True:
                ret, g_frame[peer - 1] = self.cap.read()
                if ret:
                    global predEngage
                    #print(f'peer:{peer - 1} and pred: {predEngage[peer - 1]}')
                    if predEngage[peer - 1] is -1: #undefined
                        l_frame = g_frame[peer - 1]
                    elif predEngage[peer - 1] is 0 or predEngage[peer - 1] is 1: #not engaged
                        l_frame = cv2.addWeighted(self.redImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (640, 480), (0, 0, 255), 20)
                        cv2.putText(l_frame, f'Engagement: {LABEL_VAL}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2,
                                cv2.LINE_AA)
                        cv2.putText(l_frame, f'Prediction: {predEngage[peer - 1]}', (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    elif predEngage[peer - 1] is 2: #neutral
                        l_frame = cv2.addWeighted(self.neutralImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (640, 480), (160, 172, 203), 20)
                        cv2.putText(l_frame, f'Engagement: {LABEL_VAL}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2,
                                cv2.LINE_AA)
                        cv2.putText(l_frame, f'Prediction: {predEngage[peer - 1]}', (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                        else:
                            cv2.putText(l_frame, f'Engagement: {predEngage[peer - 1]} / Cor: {self.labeledEngagement}', (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                                cv2.LINE_AA)


                    ret, buffer = cv2.imencode('.jpg', l_frame)
                    l_frame = buffer.tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + l_frame + b'\r\n')
                    isStartAudio=True
                else:
                    break

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
    wait(lambda: loadingComplete, timeout_seconds=120, waiting_for="video process ready")
    cv2.waitKey(200)
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
    global loadingComplete
    wait(lambda: loadingComplete, timeout_seconds=120, waiting_for="video process ready")
    return Response(justWebCAM(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/mfccend', methods=['POST'])
def mfccend():
    # microphone_checker_stream.process_stop()
    # mfcc_ctrl()
    return render_template('temp.html', value=0)


@app.route('/colorStat_feed/<peer>')
def colorStat_feed(peer):
    global colorStatus

    return jsonify({
        'colorStatus': str(colorStatus[peer - 1])
    })

@app.route('/generalStat_feed/<peer>')
def generalStat_feed(peer):
    global generalStatus

    returnVal = [0, 0, 0]
    if generalStatus[peer - 1][0] is True:
        returnVal[0] = 1
    if generalStatus[peer - 1][1] is True:
        returnVal[1] = 1
    if generalStatus[peer - 1][2] is True:
        returnVal[2] = 1

    return jsonify({
        'away': str(returnVal[0]),
        'smallhead': str(returnVal[1]),
        'silence': str(returnVal[2])
    })


@app.route('/vad_feed', methods=['POST'])
def vad_feed():
    # global silence
    # global size
    # global sc1
    # global sc2
    # global sc3
    # global sc4
    sc1 = VoiceActivityDetection.setSC1()
    sc2 = VoiceActivityDetection.setSC2()
    sc3 = VoiceActivityDetection.setSC3()
    sc4 = VoiceActivityDetection.setSC4()
    return jsonify({
        'SpeechCount1': int(sc1),
        'SpeechCount2': int(sc2),
        'SpeechCount3': int(sc3),
        'SpeechCount4': int(sc4)
    })
import base64
@app.route('/vad_img_feed', methods=['POST'])
# @app.route('/vad_img_feed')
def vad_img_feed():
    vc1,sc1 = VoiceActivityDetection.setVADdata(1)
    vc2,sc2 = VoiceActivityDetection.setVADdata(2)
    vc3,sc3 = VoiceActivityDetection.setVADdata(3)
    vc4,sc4 = VoiceActivityDetection.setVADdata(4)
    
    # plt.ylim([0,1])
    # plt.xticks([])
    # plt.axhline(y=0.7, color='r')
    fig, axs = plt.subplots(4)

    axs[0].plot(vc1)
    axs[0].set_ylim([0,0.7])
    axs[0].axhline(y=0.5, color='r')
    axs[1].plot(vc2)
    axs[1].set_ylim([0,0.7])
    axs[1].axhline(y=0.5, color='r')
    axs[2].plot(vc3)
    axs[2].set_ylim([0,0.7])
    axs[2].axhline(y=0.5, color='r')
    axs[3].plot(vc4)
    axs[3].set_ylim([0,0.7])
    axs[3].axhline(y=0.5, color='r')
    imgByte = BytesIO()
    # plt.show()

    plt.savefig(imgByte, format='png', bbox_inches='tight', dpi=200)
    # global plotImg
    # plotImg = imgBytes
    imgByte.seek(0)
    # encoded_string = base64.b64encode(imgByte.read())

    return send_file(imgByte, mimetype='image/png')
    # return Response(getVADdata(),mimetype='image/png')

# @app.route('/fig1')
# def fig1():
#     global img1
#     img1 = VoiceActivityDetection.getPlot(1)
#     return send_file(img1, mimetype='image/png')

# @app.route('/fig2')
# def fig2():
#     global img2
#     img2 = VoiceActivityDetection.getPlot(2)
#     return send_file(img2, mimetype='image/png')

# @app.route('/fig3')
# def fig3():
#     global img3
#     img3 = VoiceActivityDetection.getPlot(3)
#     return send_file(img3, mimetype='image/png')

# @app.route('/fig4')
# def fig4():
#     global img4
#     img4 = VoiceActivityDetection.getPlot(4)
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