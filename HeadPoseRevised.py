import cv2
import numpy as np
from whenet import WHENet
import os
from math import cos, sin
from YOLO.yolo_postprocess import YOLO
from PIL import Image
import time

def draw_axis(img, yaw, pitch, roll, horizAvg, vertiAvg, rollAvg, headArea, areaAvg, conType, tdx=None, tdy=None, size = 100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    red = (0,0,255)
    green = (0,255,0)
    blue = (255,0,0)

    if tdx == None or tdy == None:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X
    xAxis_x = size * (cos(yaw) * cos(roll)) + tdx
    xAxis_y = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y
    yAxis_x = size * (-cos(yaw) * sin(roll)) + tdx
    yAxis_y = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z
    zAxis_x = size * (sin(yaw)) + tdx
    zAxis_y = size * (-cos(yaw) * sin(pitch)) + tdy

    #위 값들: 성분벡터, 이를 통해 선 그리기
    cv2.line(img, (int(tdx), int(tdy)), (int(xAxis_x),int(xAxis_y)),red,2)
    cv2.line(img, (int(tdx), int(tdy)), (int(yAxis_x),int(yAxis_y)),green,2)
    cv2.line(img, (int(tdx), int(tdy)), (int(zAxis_x),int(zAxis_y)),blue,2)

    horizMove = int(tdx) - int(zAxis_x)
    vertiMove = int(tdy) - int(zAxis_y)
    rollByXPos = int(tdx) - int(yAxis_x)

    cv2.putText(img, 'Horiz: ' + str(horizMove), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, red, 2, cv2.LINE_AA)
    cv2.putText(img, 'Verti: ' + str(vertiMove), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, red, 2, cv2.LINE_AA)
    cv2.putText(img, 'Roll: ' + str(rollByXPos), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.1, red, 2, cv2.LINE_AA)

    if horizAvg == None or vertiAvg == None or rollAvg == None or areaAvg == None:
        cv2.putText(img, '!Calibrationing!', (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, green, 2,
                    cv2.LINE_AA)
    else:
        if abs(horizAvg - horizMove) > 80:
            conType[0] = True
            cv2.putText(img, 'horizMoveDetected', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.1, green, 2,
                        cv2.LINE_AA)
        else:
            conType[0] = False

        if vertiMove - vertiAvg > 50:
            conType[1] = True
            cv2.putText(img, 'vertiMoveDetected', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.1, green, 2,
                        cv2.LINE_AA)
        else:
            conType[1] = False

        if abs(rollAvg - rollByXPos) > 85 and headArea > areaAvg * 0.6:
            conType[2] = True
            cv2.putText(img, 'rollDetected', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.1, green, 2,
                        cv2.LINE_AA)
        else:
            conType[2] = False

        #print('area:' + str(headArea) + '/avg:' + str(areaAvg))
        if headArea < areaAvg * 0.7:
            conType[3] = True
            cv2.putText(img, 'headAreaDetected', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.1, green, 2,
                        cv2.LINE_AA)
        else:
            conType[3] = False

    return horizMove, vertiMove, rollByXPos

def process_detection(model, img, bbox, horizAvg, vertiAvg, rollAvg, areaAvg, conType):
    yMin, xMin, yMax, xMax = bbox

    yMin = max(0, yMin - abs(yMin - yMax) / 10)
    yMax = min(img.shape[0], yMax + abs(yMin - yMax) / 10)

    xMin = max(0, xMin - abs(xMin - xMax) / 5)
    xMax = min(img.shape[1], xMax + abs(xMin - xMax) / 5)
    xMax = min(xMax, img.shape[1])

    rgbImg = img[int(yMin):int(yMax), int(xMin):int(xMax)]
    rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
    rgbImg = cv2.resize(rgbImg, (224, 224))
    rgbImg = np.expand_dims(rgbImg, axis=0)

    cv2.rectangle(img, (int(xMin), int(yMin)), (int(xMax), int(yMax)), (0,0,0), 2)
    yaw, pitch, roll = model.get_angle(rgbImg)
    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])

    cv2.putText(img, 'x: '+str(int(xMax-xMin))+'/y: '+str(int(yMax-yMin)), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2,
                cv2.LINE_AA)

    headArea = int(xMax - xMin) * int(yMax - yMin)

    horizMove, vertiMove, rollByXPos = draw_axis(img, yaw, pitch, roll, horizAvg, vertiAvg, rollAvg, headArea, areaAvg, conType, tdx=(xMin+xMax)/2, tdy=(yMin+yMax)/2, size = abs(xMax-xMin)//2)
    return img, horizMove, vertiMove, rollByXPos, headArea

'''
if __name__ == "__main__":
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
    out = cv2.VideoWriter('headPoseOutput.avi', fourcc, 30, (frame.shape[1], frame.shape[0]))

    #조건 검사를 위한 변수들
    frCnt = 0
    horizSum = 0
    vertiSum = 0
    rollSum = 0
    areaSum = 0
    horizAvg = None
    vertiAvg = None
    rollAvg = None
    areaAvg = None

    startTime = time.time()
    prevTime = startTime
    CALITIME = 15
    
    while True:
        try:
            ret, frame = cap.read()
        except:
            break

        frCnt += 1
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime

        print('FPS: ' + str(1/(sec)))

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgbFrame)
        bboxes, scores, classes = yolo.YOLO_DetectProcess(img_pil)

        #for bbox in bboxes:
        #    frame, horizMove, vertiMove, rollByXPos = process_detection(whenet, frame, bbox)

        #얼굴 하나만 가지고 인식 (위에거는 여러개)
        frame, horizMove, vertiMove, rollByXPos, headArea = process_detection(whenet, frame, bboxes[0], horizAvg, vertiAvg, rollAvg, areaAvg)

        horizSum += horizMove
        vertiSum += vertiMove
        rollSum += rollByXPos
        areaSum += headArea

        #캘리브레이션 끝
        if time.time() - startTime < CALITIME+1 and time.time() - startTime > CALITIME:
            horizAvg = horizSum / frCnt
            vertiAvg = vertiSum / frCnt
            rollAvg = rollSum / frCnt
            areaAvg = areaSum / frCnt

        cv2.imshow('output', frame)
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
'''