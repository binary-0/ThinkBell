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
import json

import datetime
from flask import Flask, render_template, Response, jsonify
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    now = datetime.datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    templateData = {
            'title':'Image Streaming',
            'time': timeString
    }

    global silence
    global size
    global logStartTime
    global logEndTime
    global logType

    logStartTime = 0
    logEndTime = 0
    logType = 0
    silence = 1
    size = 1
    
    return render_template('index.html', **templateData)

@app.route('/mfccstart', methods=['POST'])
def mfccstart():
    th = threading.Thread(target=mfcc_ctrl, args=())
    th.start()

    return render_template('temp.html', value=0)

def mfcc_ctrl():
    global silence
    global size

    silence, size = microphone_checker_stream.mfcc_process()

#if __name__ == "__main__":
def gen_frames():
    whenet = WHENet(snapshot='WHENet.h5')
    yolo = YOLO()

    src = 'WIN_20210707_14_50_03_Pro.mp4'  # 웹캠이 없다면 파일 경로 넣기
    cap = cv2.VideoCapture(src)

    if src == 0:
        print('using web cam')
    else:
        print('using video, path: {}'.format(src))

    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('Output.avi', fourcc, 30, (frame.shape[1], frame.shape[0]))

    # 조건 검사를 위한 변수들
    frCnt = 0
    horizSum = 0
    vertiSum = 0
    rollSum = 0
    areaSum = 0
    EARSum = 0
    horizAvg = None
    vertiAvg = None
    rollAvg = None
    areaAvg = None
    EARAvg = None

    conType = [False, False, False, False]

    startTime = time.time()
    prevTime = startTime
    CALITIME = 15
    TIMETHRESHOLD = 3
    EARTime = None
    HeadTime = None
    FaceTime = None

    global logStartTime
    global logEndTime
    global logType

    logStartTime = 0
    logEndTime = 0
    logType = 0
    
    frameTemp = 0
    frameCtrl = None
    frameBack = 0

    while True:
        frCnt += 1

        if frameCtrl is None:
            frameBack = frCnt
        else:
            frameBack = frameCtrl
            frCnt = frameCtrl
            print('move back')
            frameCtrl = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frameBack)

        try:
            ret, frame = cap.read()
            print(frameBack)
        except:
            break

        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

        global returnCheck
        if returnCheck == 1:
            #logStartTime = 0
            #logEndTime = 0
            #logType = 0
            returnCheck = 0

        print('FPS: ' + str(1 / (sec)))

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgbFrame)
        bboxes, scores, classes = yolo.YOLO_DetectProcess(img_pil)

        bgr = cv2.flip(frame, 1)

        # for bbox in bboxes:
        #    frame, horizMove, vertiMove, rollByXPos = process_detection(whenet, frame, bbox)

        # 얼굴 하나만 가지고 인식 (위에거는 여러개)
        if len(bboxes) == 0:
            continue
        frame, horizMove, vertiMove, rollByXPos, headArea = process_detection(whenet, frame, bboxes[0], horizAvg, vertiAvg,
                                                                              rollAvg, areaAvg, conType)

        EAR = calculate_ear(rgbFrame, draw=frame)

        # 캘리브레이션 끝
        if time.time() - startTime < CALITIME + 1 and time.time() - startTime > CALITIME:
            horizAvg = horizSum / frCnt
            vertiAvg = vertiSum / frCnt
            rollAvg = rollSum / frCnt
            areaAvg = areaSum / frCnt
            EARAvg = EARSum / frCnt
        else:
            horizSum += horizMove
            vertiSum += vertiMove
            rollSum += rollByXPos
            areaSum += headArea

            if EAR is not None:
                EARSum += EAR
        

        #EAR Threshold는 Main에서
        if EARAvg is not None and EAR is not None:
            cv2.putText(frame, f'EAR:{EAR}', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            earCurTime = time.time()
            if EAR < EARAvg*0.9:
                if EARTime is None:
                    EARTime = earCurTime
                    frameTemp = frCnt
                else:
                    if earCurTime - EARTime > TIMETHRESHOLD:
                        cv2.putText(frame, 'EARDetected', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2,
                                    cv2.LINE_AA)
            else:
                if EARTime is not None and time.time() - EARTime > TIMETHRESHOLD:
                    logStartTime = EARTime - startTime
                    logEndTime = time.time() - startTime
                    logType = 0
                EARTime = None

        if conType[0] == True or conType[1] == True or conType[2] == True:
            if HeadTime is None:
                HeadTime = time.time()
                frameTemp = frCnt
            else:
                headCurTime = time.time()
                if headCurTime - HeadTime> TIMETHRESHOLD:
                    pass
        else:
            if HeadTime is not None and time.time() - HeadTime > TIMETHRESHOLD:
                logStartTime = HeadTime - startTime
                logEndTime = time.time() - startTime
                logType = 1
                #frameCtrl = frameTemp
            HeadTime = None

        if conType[3] == True:
            if FaceTime is None:
                FaceTime = time.time()
                frameTemp = frCnt
            else:
                faceCurTime = time.time()
                if faceCurTime - FaceTime > TIMETHRESHOLD:
                    pass
        else:
            if FaceTime is not None and time.time() - FaceTime > TIMETHRESHOLD:
                logStartTime = FaceTime - startTime
                logEndTime = time.time() - startTime
                logType = 2
            FaceTime = None


        #cv2.imshow('output', frame)
        out.write(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    #cap.release()
    #cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
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

@app.route('/log_feed', methods=['POST'])
def log_feed():
    global logStartTime
    global logEndTime
    global logType
    global returnCheck

    returnCheck = 1
    return jsonify({
        'startTime': str(logStartTime),
        'endTime': str(logEndTime),
        'behaviorType': str(logType)
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1')