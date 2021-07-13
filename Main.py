import threading

from matplotlib.pyplot import tricontour

import cv2
import numpy as np
from whenet import WHENet
import os
from math import cos, sin
from YOLO.yolo_postprocess import YOLO
from PIL import Image
import time
from HeadPoseRevised import process_detection
from EAR import calculate_ear
import microphone_checker_stream
from waiting import wait
import json
from queue import Queue

import datetime
from flask import Flask, render_template, Response, jsonify, send_file
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    now = datetime.datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    templateData = {
            'title': 'Image Streaming',
            'time': timeString
    }

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
    global img
    global logStudentName

    g_frame = None

    logStudentName = 0
    logStartTime = 0
    logEndTime = 0
    logType = 0
    silence = 1
    size = 1
    isVideoSystemReady = False
    isLiveLocal = 1 #Local
    isRealDebug = 0 #Debug
    returnCheck = 0
    loadingComplete = False
    img = None

    global isFrameReady
    isFrameReady = False

    return render_template('index.html', **templateData)

@app.route('/mfccstart', methods=['POST'])
def mfccstart():
    th = threading.Thread(target=mfcc_ctrl, args=())
    th.start()

    return render_template('temp.html', value=0)

def mfcc_ctrl():
    global silence
    global size
    global img

    silence, size, img = microphone_checker_stream.mfcc_process()

def real_gen_frames():
    whenet = WHENet(snapshot='WHENet.h5')
    yolo = YOLO()

    global loadingComplete
    loadingComplete = True

    global isLiveLocal
    global g_frame
    peerNum = 4

    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # out = cv2.VideoWriter('Output.avi', fourcc, 6, (frame.shape[1], frame.shape[0]))

    # 조건 검사를 위한 변수들
    frCnt = 0
    horizSum = [0, 0, 0, 0]
    vertiSum = [0, 0, 0, 0]
    rollSum = [0, 0, 0, 0]
    areaSum = [0, 0, 0, 0]
    EARSum = [0, 0, 0, 0]
    horizAvg = [None, None, None, None]
    vertiAvg = [None, None, None, None]
    rollAvg = [None, None, None, None]
    areaAvg = [None, None, None, None]
    EARAvg = [None, None, None, None]

    conType = list(range(peerNum))
    for i in range(0, peerNum):
        conType[i] = [False, False, False, False]

    startTime = time.time()
    prevTime = startTime
    CALITIME = 15
    TIMETHRESHOLD = 3
    EARTime = [None, None, None, None]
    HeadTime = [None, None, None, None]
    FaceTime = [None, None, None, None]

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
    dQ = [detectionQueue(), detectionQueue(), detectionQueue(), detectionQueue()]

    time.sleep(3)
    while True:
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

        if g_frame is None:
            continue

        try:
            if round(1 / sec) is not 0:
                frCnt += round(30 / round(1 / sec))
            else:
                frCnt += round(30 / (1 / sec))
        except:
            frCnt += 1
        try:
            print('FPS: ' + str(1 / (sec)))
        except:
            print('FPS___')

        if frameCtrl is None:
            frameBack = frCnt
        else:
            frameBack = frameCtrl
            frCnt = frameCtrl
            print('move back')
            frameCtrl = None

        #for i in range(0, peerNum):
        #    cap[i].set(cv2.CAP_PROP_POS_FRAMES, frameBack)

        # try:
        #     for i in range(0, peerNum):
        #         ret[i], frame[i] = cap[i].read()
        #     # print(frameBack)
        # except:
        #     break
        # if frame[0] is None or frame[1] is None:
        #     break

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
        rgbFrame = list(range(peerNum))
        for i in range(0, peerNum):
            rgbFrame[i] = cv2.cvtColor(sendedFrame[i], cv2.COLOR_BGR2RGB)
        img_pil = list(range(peerNum))
        for i in range(0, peerNum):
            img_pil[i] = Image.fromarray(rgbFrame[i])

        bboxes = list(range(peerNum))
        scores = list(range(peerNum))
        classes = list(range(peerNum))
        for i in range(0, peerNum):
            bboxes[i], scores[i], classes[i] = yolo.YOLO_DetectProcess(img_pil[i])

        # bgr = cv2.flip(frame, 1)

        # for bbox in bboxes:
        #    frame, horizMove, vertiMove, rollByXPos = process_detection(whenet, frame, bbox)

        # 얼굴 하나만 가지고 인식 (위에거는 여러개)
        if len(bboxes[0]) is 0 or len(bboxes[1]) is 0:
            continue

        horizMove = list(range(peerNum))
        vertiMove = list(range(peerNum))
        rollByXPos = list(range(peerNum))
        headArea = list(range(peerNum))
        l_frame = list(range(peerNum))
        for i in range(0, peerNum):
            l_frame[i], horizMove[i], vertiMove[i], rollByXPos[i], headArea[i] = process_detection(whenet, sendedFrame[i],
                                                                                                 bboxes[i][0],
                                                                                                 horizAvg[i],
                                                                                                 vertiAvg[i],
                                                                                                 rollAvg[i], areaAvg[i],
                                                                                                 conType[i])
        EAR = list(range(peerNum))
        for i in range(0, peerNum):
            EAR[i] = calculate_ear(rgbFrame[i], draw=l_frame[i])

        # 캘리브레이션 끝
        if time.time() - startTime < CALITIME + 1 and time.time() - startTime > CALITIME:
            for i in range(0, peerNum):
                horizAvg[i] = horizSum[i] / frCnt
                vertiAvg[i] = vertiSum[i] / frCnt
                rollAvg[i] = rollSum[i] / frCnt
                areaAvg[i] = areaSum[i] / frCnt
                EARAvg[i] = EARSum[i] / frCnt
        else:
            for i in range(0, peerNum):
                horizSum[i] += horizMove[i]
                vertiSum[i] += vertiMove[i]
                rollSum[i] += rollByXPos[i]
                areaSum[i] += headArea[i]

                if EAR[i] is not None:
                    EARSum[i] += EAR[i]

        # EAR Threshold는 Main에서
        for i in range(0, peerNum):
            if EARAvg[i] is not None and EAR[i] is not None:
                cv2.putText(l_frame[i], f'EAR:{EAR[i]}', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                earCurTime = time.time()
                if EAR[i] < EARAvg[i] * 0.9:
                    isDetectedOnce[i] = 1
                    if EARTime[i] is None:
                        EARTime[i] = earCurTime
                        frameTemp = frCnt
                    else:
                        if earCurTime - EARTime[i] > TIMETHRESHOLD:
                            cv2.putText(l_frame[i], 'EARDetected', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0),
                                        2,
                                        cv2.LINE_AA)
                else:
                    if EARTime[i] is not None and time.time() - EARTime[i] > TIMETHRESHOLD:
                        logStudentName = i
                        logStartTime = EARTime[i] - startTime
                        logEndTime = time.time() - startTime
                        logType = 0
                    EARTime[i] = None

        for i in range(0, peerNum):
            if conType[i][0] == True or conType[i][1] == True or conType[i][2] == True:
                isDetectedOnce[i] = 1
                if HeadTime[i] is None:
                    HeadTime[i] = time.time()
                    frameTemp = frCnt
                else:
                    headCurTime = time.time()
                    if headCurTime - HeadTime[i] > TIMETHRESHOLD:
                        pass
            else:
                if HeadTime[i] is not None and time.time() - HeadTime[i] > TIMETHRESHOLD:
                    logStudentName = i
                    logStartTime = HeadTime[i] - startTime
                    logEndTime = time.time() - startTime
                    logType = 1
                    # frameCtrl = frameTemp
                HeadTime[i] = None

        for i in range(0, peerNum):
            if conType[i][3] == True:
                isDetectedOnce[i] = 1
                if FaceTime[i] is None:
                    FaceTime[i] = time.time()
                    frameTemp = frCnt
                else:
                    faceCurTime = time.time()
                    if faceCurTime - FaceTime[i] > TIMETHRESHOLD:
                        pass
            else:
                if FaceTime[i] is not None and time.time() - FaceTime[i] > TIMETHRESHOLD:
                    logStudentName = i
                    logStartTime = FaceTime[i] - startTime
                    logEndTime = time.time() - startTime
                    logType = 2
                FaceTime[i] = None

        for i in range(0, peerNum):
            global DetectRet
            DetectRet[i] = dQ[i].detectionPush(isDetectedOnce[i])

        # cv2.imshow('output', frame)
        # out.write(frame)
        #global frameReady
        #for i in range(0, peerNum):
        #    frameReady[i] = True

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

class detectionQueue:
    def __init__(self):
        self.que = Queue()
        self.size = 0
        self.sum = 0
        self.avg = 0

        self.yelloThresh = 4
        self.redThresh = 6
        self.maxSize = 10

        global DetectRet
        DetectRet = list(range(4)) #peerNum

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
    gen_frame_thread = threading.Thread(target=real_gen_frames)
    gen_frame_thread.start()
    webCam_thread = threading.Thread(target=justWebCAM)
    webCam_thread.start()

    def __init__(self, peer):
        global isLiveLocal
        global frameReady
        #global loadingComplete
        global g_frame
        self.peerNum = 4

        g_frame = list(range(self.peerNum))
        frameReady = list(range(self.peerNum))

        self.redImg = np.full((720, 1280, 3), (0, 0, 255), dtype=np.uint8)
        self.greenImg = np.full((720, 1280, 3), (0, 255, 0), dtype=np.uint8)
        self.yelloImg = np.full((720, 1280, 3), (0, 255, 255), dtype=np.uint8)

        if isLiveLocal is 0:
            self.srcPath = 0
        else:
            self.srcPath = f'./SampleVideo{peer}.mp4'
        self.cap = cv2.VideoCapture(self.srcPath)
        #wait(lambda: loadingComplete, timeout_seconds=120, waiting_for="video process ready")


    def local_frames(self, peer):
        global g_frame
        global isVideoSystemReady
        global isLiveLocal
        global isRealDebug
        global frameReady

        peer = int(peer)
        ret, g_frame[peer-1] = self.cap.read()

        #wait(lambda: frameReady[peer-1], timeout_seconds=120, waiting_for="video process ready")

        if self.cap.isOpened():
            while True:
                ret, g_frame[peer-1] = self.cap.read()
                if ret:
                    global DetectRet
                    if DetectRet[peer-1] is 1:
                        l_frame = cv2.addWeighted(self.greenImg, 0.15, g_frame[peer-1], 0.85, 0)
                    elif DetectRet[peer-1] is 2:
                        l_frame = cv2.addWeighted(self.yelloImg, 0.15, g_frame[peer - 1], 0.85, 0)
                    elif DetectRet[peer-1] is 3:
                        l_frame = cv2.addWeighted(self.redImg, 0.15, g_frame[peer - 1], 0.85, 0)
                    else:
                        l_frame = g_frame[peer-1]
                    ret, buffer = cv2.imencode('.jpg', l_frame)
                    l_frame = buffer.tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + l_frame + b'\r\n')
                else:
                    break

                if isLiveLocal is 1:
                    if cv2.waitKey(15) & 0xFF == ord('q'):  # press q to quit
                        break

        else:
            print('Cannot Open File ERR')


# def frameSession1():
#     global byframe
#     global isRealDebug
#
#     try:
#         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + byframe[0] + b'\r\n')
#     except:
#         pass
#
# def frameSession2():
#     global byframe
#     try:
#         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + byframe[1] + b'\r\n')
#     except:
#         pass


@app.route('/video_feed/<peer>')
def video_feed(peer):
    """Video streaming route. Put this in the src attribute of an img tag."""
    global isRealDebug
    global frameReady
    global loadingComplete
    wait(lambda: loadingComplete, timeout_seconds=120, waiting_for="video process ready")
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
    microphone_checker_stream.process_stop()
    return render_template('temp.html', value=0)

@app.route('/mfcc_feed', methods=['POST'])
def mfcc_feed():
    global silence
    global size

    return jsonify({
        'silence': str(silence),
        'size': str(size),
        'rate': str(100 - silence / size * 100),
    })

@app.route('/fig')
def fig():
    global img
    print(type(img))
    return send_file(img, mimetype='image/png')

@app.route('/log_feed', methods=['POST'])
def log_feed():
    global logStartTime
    global logEndTime
    global logType
    global returnCheck
    global logStudentName

    if logStartTime is not 0 and logEndTime is not 0 and logType is not 0:
        returnCheck = 1

    return jsonify({
        'name': str(logStudentName),
        'startTime': str(logStartTime),
        'endTime': str(logEndTime),
        'behaviorType': str(logType)
    })

def debug_gen_frames():
    whenet = WHENet(snapshot='WHENet.h5')
    yolo = YOLO()

    global isLiveLocal
    peerNum = 4

    cap = list(range(peerNum))
    ret = list(range(peerNum))
    frame = list(range(peerNum))

    if isLiveLocal is 0:
        cap[0] = cv2.VideoCapture(0)
    else:  # Local video
        cap[0] = cv2.VideoCapture('./SampleVideo.mp4')
        cap[1] = cv2.VideoCapture('./SampleVideo2.mp4')

    for i in range(0, peerNum):
        ret[i], frame[i] = cap[i].read()
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('Output.avi', fourcc, 6, (frame.shape[1], frame.shape[0]))

    # 조건 검사를 위한 변수들
    frCnt = 0
    horizSum = [0, 0]
    vertiSum = [0, 0]
    rollSum = [0, 0]
    areaSum = [0, 0]
    EARSum = [0, 0]
    horizAvg = [None, None]
    vertiAvg = [None, None]
    rollAvg = [None, None]
    areaAvg = [None, None]
    EARAvg = [None, None]

    conType = list(range(peerNum))
    for i in range(0, peerNum):
        conType[i] = [False, False, False, False]

    startTime = time.time()
    prevTime = startTime
    CALITIME = 15
    TIMETHRESHOLD = 3
    EARTime = [None, None]
    HeadTime = [None, None]
    FaceTime = [None, None]

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

    while True:
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

        try:
            if round(1 / sec) is not 0:
                frCnt += round(30 / round(1 / sec))
            else:
                frCnt += round(30 / (1 / sec))
        except:
            frCnt += 1
        try:
            print('FPS: ' + str(1 / (sec)))
        except:
            print('FPS___')

        if frameCtrl is None:
            frameBack = frCnt
        else:
            frameBack = frameCtrl
            frCnt = frameCtrl
            print('move back')
            frameCtrl = None

        for i in range(0, peerNum):
            cap[i].set(cv2.CAP_PROP_POS_FRAMES, frameBack)

        try:
            for i in range(0, peerNum):
                ret[i], frame[i] = cap[i].read()
            #print(frameBack)
        except:
            break
        if frame[0] is None or frame[1] is None:
            break

        global returnCheck
        if returnCheck == 1:
            logStudentName = 0
            logStartTime = 0
            logEndTime = 0
            logType = 0
            returnCheck = 0

        rgbFrame = list(range(peerNum))
        for i in range(0, peerNum):
            rgbFrame[i] = cv2.cvtColor(frame[i], cv2.COLOR_BGR2RGB)
        img_pil = list(range(peerNum))
        for i in range(0, peerNum):
            img_pil[i] = Image.fromarray(rgbFrame[i])

        bboxes = list(range(peerNum))
        scores = list(range(peerNum))
        classes = list(range(peerNum))
        for i in range(0, peerNum):
            bboxes[i], scores[i], classes[i] = yolo.YOLO_DetectProcess(img_pil[i])

        #bgr = cv2.flip(frame, 1)

        # for bbox in bboxes:
        #    frame, horizMove, vertiMove, rollByXPos = process_detection(whenet, frame, bbox)

        # 얼굴 하나만 가지고 인식 (위에거는 여러개)
        if len(bboxes[0]) is 0 or len(bboxes[1]) is 0:
            continue

        horizMove = list(range(peerNum))
        vertiMove = list(range(peerNum))
        rollByXPos = list(range(peerNum))
        headArea = list(range(peerNum))
        for i in range(0, peerNum):
            frame[i], horizMove[i], vertiMove[i], rollByXPos[i], headArea[i] = process_detection(whenet, frame[i], bboxes[i][0], horizAvg[i], vertiAvg[i],
                                                                                                 rollAvg[i], areaAvg[i], conType[i])
        EAR = list(range(peerNum))
        for i in range(0, peerNum):
            EAR[i] = calculate_ear(rgbFrame[i], draw=frame[i])

        # 캘리브레이션 끝
        if time.time() - startTime < CALITIME + 1 and time.time() - startTime > CALITIME:
            for i in range(0, peerNum):
                horizAvg[i] = horizSum[i] / frCnt
                vertiAvg[i] = vertiSum[i] / frCnt
                rollAvg[i] = rollSum[i] / frCnt
                areaAvg[i] = areaSum[i] / frCnt
                EARAvg[i] = EARSum[i] / frCnt
        else:
            for i in range(0, peerNum):
                horizSum[i] += horizMove[i]
                vertiSum[i] += vertiMove[i]
                rollSum[i] += rollByXPos[i]
                areaSum[i] += headArea[i]

                if EAR[i] is not None:
                    EARSum[i] += EAR[i]

        # EAR Threshold는 Main에서
        for i in range(0, peerNum):
            if EARAvg[i] is not None and EAR[i] is not None:
                cv2.putText(frame[i], f'EAR:{EAR[i]}', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                earCurTime = time.time()
                if EAR[i] < EARAvg[i] * 0.9:
                    if EARTime[i] is None:
                        EARTime[i] = earCurTime
                        frameTemp = frCnt
                    else:
                        if earCurTime - EARTime[i] > TIMETHRESHOLD:
                            cv2.putText(frame[i], 'EARDetected', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2,
                                        cv2.LINE_AA)
                else:
                    if EARTime[i] is not None and time.time() - EARTime[i] > TIMETHRESHOLD:
                        logStartTime = EARTime[i] - startTime
                        logEndTime = time.time() - startTime
                        logType = 0
                    EARTime[i] = None

        for i in range(0, peerNum):
            if conType[i][0] == True or conType[i][1] == True or conType[i][2] == True:
                if HeadTime[i] is None:
                    HeadTime[i] = time.time()
                    frameTemp = frCnt
                else:
                    headCurTime = time.time()
                    if headCurTime - HeadTime[i] > TIMETHRESHOLD:
                        pass
            else:
                if HeadTime[i] is not None and time.time() - HeadTime[i] > TIMETHRESHOLD:
                    logStartTime = HeadTime[i] - startTime
                    logEndTime = time.time() - startTime
                    logType = 1
                    # frameCtrl = frameTemp
                HeadTime[i] = None

        for i in range(0, peerNum):
            if conType[i][3] == True:
                if FaceTime[i] is None:
                    FaceTime[i] = time.time()
                    frameTemp = frCnt
                else:
                    faceCurTime = time.time()
                    if faceCurTime - FaceTime[i] > TIMETHRESHOLD:
                        pass
            else:
                if FaceTime[i] is not None and time.time() - FaceTime[i] > TIMETHRESHOLD:
                    logStartTime = FaceTime[i] - startTime
                    logEndTime = time.time() - startTime
                    logType = 2
                FaceTime[i] = None

        # cv2.imshow('output', frame)
        #out.write(frame)
        buffer = list(range(peerNum))
        global byframe
        global frameReady
        byframe = list(range(peerNum))
        frameReady = list(range(peerNum))
        for i in range(0, peerNum):
            ret[i], buffer[i] = cv2.imencode('.jpg', frame[i])
            byframe[i] = buffer[i].tobytes()
            frameReady[i] = True

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + byframe[0] + b'\r\n')
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + byframe[1] + b'\r\n')

if __name__ == '__main__':
    app.run(host='127.0.0.1')