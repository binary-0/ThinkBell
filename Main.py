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

from multiprocessing import Process

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

    global predEngage
    predEngage = [-1, -1, -1, -1]

    global isStartAudio
    isStartAudio=False


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

    

    global DetectRet
    # DetectRet = list(range(4))  # peerNum
    DetectRet = [0, 0, 0, 0]

    global isFrameReady
    isFrameReady = False

    gen_frame_thread = threading.Thread(target=real_gen_frames)
    gen_frame_thread.start()
    webCam_thread = threading.Thread(target=justWebCAM)
    webCam_thread.start()

    audio_thread = threading.Thread(target=play_audio)
    audio_thread.start()

    # threading.Thread(target=vad_ctrl).start()
    Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio1.wav",)).start()
    Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio2.wav",)).start()
    Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio3.wav",)).start()
    Process(target=VoiceActivityDetection.vadStart, args=("SampleAudio4.wav",)).start()
    
    return render_template('index.html', **templateData)

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

    global loadingComplete
    loadingComplete = True

    global isLiveLocal
    global g_frame
    peerNum = 4

    # 조건 검사를 위한 변수들
    frCnt = 0

    startTime = time.time()
    prevTime = startTime
    global CALITIME
    global TIMETHRESHOLD

    print(str(TIMETHRESHOLD - 0))
    CNNTime = [None, None, None, None]

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
    while True:
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

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

        isDetectedOnce = [0, 0, 0, 0]

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

    while True:
        myRet, myFrame = myCap.read()

        if myRet:
            ret, buffer = cv2.imencode('.jpg', myFrame)
            myBytes = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + myBytes + b'\r\n')


class Streaming:
    def __init__(self, peer):
        global isLiveLocal
        global frameReady
        # global loadingComplete
        global g_frame
        self.peerNum = 4

        g_frame = list(range(self.peerNum))
        frameReady = list(range(self.peerNum))

        self.redImg = np.full((720, 1280, 3), (0, 0, 255), dtype=np.uint8)
        self.greenImg = np.full((720, 1280, 3), (0, 255, 0), dtype=np.uint8)
        self.blueImg = np.full((720, 1280, 3), (255, 0, 0), dtype=np.uint8)
        self.yelloImg = np.full((720, 1280, 3), (0, 255, 255), dtype=np.uint8)

        if isLiveLocal is 0:
            self.srcPath = 0
        else:
            self.srcPath = f'./SampleVideo{peer}.mp4'
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

        peer = int(peer)
        ret, g_frame[peer - 1] = self.cap.read()

        # wait(lambda: frameReady[peer-1], timeout_seconds=120, waiting_for="video process ready")
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
                        cv2.rectangle(l_frame, (0, 0), (1280, 720), (0, 0, 255), 20)
                        if self.labeledEngagement is -1:
                            cv2.putText(l_frame, f'Engagement: {predEngage[peer - 1]}', (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                        else:
                            cv2.putText(l_frame, f'Engagement: {predEngage[peer - 1]} / Cor: {self.labeledEngagement}', (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                    elif predEngage[peer - 1] is 2: #neutral
                        l_frame = cv2.addWeighted(self.blueImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (1280, 720), (255, 0, 0), 20)
                        if self.labeledEngagement is -1:
                            cv2.putText(l_frame, f'Engagement: {predEngage[peer - 1]}', (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                        else:
                            cv2.putText(l_frame, f'Engagement: {predEngage[peer - 1]} / Cor: {self.labeledEngagement}', (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                    else: #highly engaged
                        l_frame = cv2.addWeighted(self.greenImg, 0.1, g_frame[peer - 1], 0.9, 0)
                        cv2.rectangle(l_frame, (0, 0), (1280, 720), (0, 255, 0), 20)
                        if self.labeledEngagement is -1:
                            cv2.putText(l_frame, f'Engagement: {predEngage[peer - 1]}', (10, 100),
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


@app.route('/vad_feed', methods=['POST'])
def vad_feed():
    # global silence
    # global size
    sc1 = VoiceActivityDetection.getVADdata(1)
    sc2 = VoiceActivityDetection.getVADdata(2)
    sc3 = VoiceActivityDetection.getVADdata(3)
    sc4 = VoiceActivityDetection.getVADdata(4)
    return jsonify({
        'SpeechCount1': str(sc1),
        'SpeechCount2': str(sc2),
        'SpeechCount3': str(sc3),
        'SpeechCount4': str(sc4)
    })

@app.route('/fig1')
def fig1():
    global img1
    img1 = VoiceActivityDetection.getPlot(1)
    return send_file(img1, mimetype='image/png')

@app.route('/fig2')
def fig2():
    global img2
    img2 = VoiceActivityDetection.getPlot(2)
    return send_file(img2, mimetype='image/png')

@app.route('/fig3')
def fig3():
    global img3
    img3 = VoiceActivityDetection.getPlot(3)
    return send_file(img3, mimetype='image/png')

@app.route('/fig4')
def fig4():
    global img4
    img4 = VoiceActivityDetection.getPlot(4)
    return send_file(img4, mimetype='image/png')


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