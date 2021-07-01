import threading

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

import datetime
from flask import Flask, render_template, Response, request
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
    return render_template('index.html', **templateData)

@app.route('/mfccstart')
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

    src = 0  # 웹캠이 없다면 파일 경로 넣기
    cap = cv2.VideoCapture(src)

    if src == 0:
        print('using web cam')
    else:
        print('using video, path: {}'.format(src))

    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('headPoseOutput.avi', fourcc, 30, (frame.shape[1], frame.shape[0]))

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

    startTime = time.time()
    prevTime = startTime
    CALITIME = 15
    EARTime = None

    while True:
        try:
            ret, frame = cap.read()
        except:
            break

        frCnt += 1
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

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
                                                                              rollAvg, areaAvg)

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
            if EAR < EARAvg*0.85:
                if EARTime is None:
                    EARTime = earCurTime
                else:
                    if earCurTime - EARTime > 1.5:
                        cv2.putText(frame, 'EARDetected', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2,
                                    cv2.LINE_AA)
            else:
                EARTime = None


        #cv2.imshow('output', frame)
        #out.write(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

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

@app.route('/mfcc_feed', methods=['GET','POST'])
def mfcc_feed():
    mfccData = {
        'silence': str(silence),
        'size': str(size),
        'rate': str(100 - silence / size * 100)
    }
    return render_template('result.html', **mfccData)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)