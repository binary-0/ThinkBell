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

class FrameGenerator:
    def __init__(self):
        self.whenet = WHENet(snapshot='WHENet.h5')
        self.yolo = YOLO()

        #out = cv2.VideoWriter('headPoseOutput.avi', fourcc, 30, (frame.shape[1], frame.shape[0]))

        # 조건 검사를 위한 변수들
        self.frCnt = 0
        self.horizSum = 0
        self.vertiSum = 0
        self.rollSum = 0
        self.areaSum = 0
        self.EARSum = 0
        self.horizAvg = None
        self.vertiAvg = None
        self.rollAvg = None
        self.areaAvg = None
        self.EARAvg = None

        self.startTime = time.time()
        self.prevTime = self.startTime
        self.CALITIME = 15
        self.EARTime = None

    def gen_frames(self, inputFrame):

        self.frCnt += 1
        curTime = time.time()
        sec = curTime - self.prevTime
        self.prevTime = curTime

        print('FPS: ' + str(1 / (sec)))

        rgbFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgbFrame)
        bboxes, scores, classes = self.yolo.YOLO_DetectProcess(img_pil)

        bgr = cv2.flip(inputFrame, 1)

        # for bbox in bboxes:
        #    frame, horizMove, vertiMove, rollByXPos = process_detection(whenet, frame, bbox)

        # 얼굴 하나만 가지고 인식 (위에거는 여러개)
        if len(bboxes) == 0:
            return None

        frame, horizMove, vertiMove, rollByXPos, headArea = process_detection(self.whenet, inputFrame, bboxes[0], self.horizAvg, self.vertiAvg,
                                                                              self.rollAvg, self.areaAvg)

        EAR = calculate_ear(rgbFrame, draw=frame)

        # 캘리브레이션 끝
        if time.time() - self.startTime < self.CALITIME + 1 and time.time() - self.startTime > self.CALITIME:
            self.horizAvg = self.horizSum / self.frCnt
            self.vertiAvg = self.vertiSum / self.frCnt
            self.rollAvg = self.rollSum / self.frCnt
            self.areaAvg = self.areaSum / self.frCnt
            self.EARAvg = self.EARSum / self.frCnt
        else:
            self.horizSum += horizMove
            self.vertiSum += vertiMove
            self.rollSum += rollByXPos
            self.areaSum += headArea

            if EAR is not None:
                self.EARSum += EAR

        #EAR Threshold는 Main에서
        if self.EARAvg is not None and EAR is not None:
            cv2.putText(frame, f'EAR:{EAR}', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            earCurTime = time.time()
            if EAR < self.EARAvg*0.85:
                if self.EARTime is None:
                    EARTime = earCurTime
                else:
                    if earCurTime - self.EARTime > 1.5:
                        cv2.putText(frame, 'EARDetected', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2,
                                    cv2.LINE_AA)
            else:
                self.EARTime = None

        return frame


