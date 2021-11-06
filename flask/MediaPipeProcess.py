import mediapipe as mp
import numpy as np
import cv2

gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}

##Different Playground##
class PoseDetector:
    def __init__(self, mode = False, upBody = False, smooth=True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=False):
        #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        #print('findPoseDone')
        return img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS

    def getPosition(self, img, draw=False):
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
        self.file = np.genfromtxt('gesture_train.csv', delimiter=',')
        self.angle = self.file[:,:-1].astype(np.float32)
        self.label = self.file[:, -1].astype(np.float32)
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(self.angle, cv2.ml.ROW_SAMPLE, self.label) # 로딩에 넣어야하나
        
    def findHands(self, img, draw = True):
        #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        #print('findHandsDone')
        return img

    def getPosition(self, img, handNo = 0, draw = True):
        lmlist = []
        idx = -1
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        
        if self.results.multi_hand_landmarks:
            rps_result = []
            
            for res in self.results.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                
                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree
                
                # Inference gesture
                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = self.knn.findNearest(data, 3)
                idx = int(results[0][0])

                # # Draw gesture result
                # if idx in gesture.keys():
                #     #org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                #     print(gesture[idx].upper())
                #     # cv2.putText(img, text=gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)                

        return lmlist, idx
###Different Playground###

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def calculate_single_ear(eye_landmarks_np):
    upper_mean = np.mean(eye_landmarks_np[0:7], axis=0)
    lower_mean = np.mean(eye_landmarks_np[7:14], axis=0)
    return euclidean_distance(upper_mean, lower_mean) / euclidean_distance(eye_landmarks_np[14], eye_landmarks_np[15])

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def landmarks_to_np(landmarks):
    landmarks = [[landmark.x, landmark.y] for landmark in landmarks]
    return np.array(landmarks)

DRAWING_SPEC = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
FACE_MESH = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
LEFT_EYE_INDICES = [466, 388, 387, 386, 385, 384, 398, 249, 390, 373, 374, 380, 381, 382, 263, 362]
RIGHT_EYE_INDICES = [246, 161, 160, 159, 158, 157, 173, 7, 163, 144, 145, 153, 154, 155, 33, 133]

poseDetector = PoseDetector()
handDetector = HandDetector()

def mediapipe_process(rgb, oriImg, TPImg):
    ret_EAR = -1
    ret_Area = -1
    ret_HeadPose = [-1, -1]
    ret_hand = [[], []]
    ret_pose = None

    rgb.flags.writeable = False
    
    handDetector.findHands(oriImg, False)
    poseDetector.findPose(oriImg, False)
    
    results = FACE_MESH.process(rgb)

    rgb.flags.writeable = True

    #HP
    img_h, img_w, img_c = rgb.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        #print(f'LEN:{len(results.multi_face_landmarks)}')
        if len(results.multi_face_landmarks) is None:
            ret_EAR = -1
            ret_Area = -1
        
        face_landmarks = results.multi_face_landmarks[0]
        left_eye_landmarks_np = landmarks_to_np([face_landmarks.landmark[i] for i in LEFT_EYE_INDICES])
        left_ear = calculate_single_ear(left_eye_landmarks_np)

        right_eye_landmarks_np = landmarks_to_np([face_landmarks.landmark[i] for i in RIGHT_EYE_INDICES])
        right_ear = calculate_single_ear(right_eye_landmarks_np)

        h, w, c = rgb.shape
        cx_min=  w
        cy_min = h
        cx_max= cy_max= 0

        for id, lm in enumerate(face_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if cx<cx_min:
                cx_min=cx
            if cy<cy_min:
                cy_min=cy
            if cx>cx_max:
                cx_max=cx
            if cy>cy_max:
                cy_max=cy
        ret_EAR = (left_ear + right_ear) / 2
        ret_Area = int((cx_max - cx_min)*(cy_max - cy_min))

        #HP
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])       
        
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w

        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        rmat, jac = cv2.Rodrigues(rot_vec)

        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        y = angles[0] * 360
        x = angles[1] * 360

        #text = f"x:{round(x, 2)}, y:{round(y, 2)} / "
        #0: forward
        #1: horizontal movement
        #2: vertical(upper) movement
        ret_HeadPose = [x, y]
    try:
        ret_pose = poseDetector.getPosition(TPImg)
    except:
        ret_pose = []
    try:
        ret_hand[0], handGesIdx1 = handDetector.getPosition(TPImg, 0, draw=False)
    except:
        ret_hand[0] = []
        
    try:
        ret_hand[1], handGesIdx2 = handDetector.getPosition(TPImg, 1, draw=False)
        if handGesIdx1 == -1:
            handGesIdx1 = handGesIdx2
    except:
        ret_hand[1] = []

    if handGesIdx1 is 2:
        handGesIdx1 = 9
    elif handGesIdx1 is 0 or handGesIdx1 is 7:
        handGesIdx1 = 5

    return ret_EAR, ret_Area, ret_HeadPose, ret_hand, ret_pose, handGesIdx1