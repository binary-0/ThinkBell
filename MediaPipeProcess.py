import mediapipe as mp
import numpy as np

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
FACE_MESH = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
LEFT_EYE_INDICES = [466, 388, 387, 386, 385, 384, 398, 249, 390, 373, 374, 380, 381, 382, 263, 362]
RIGHT_EYE_INDICES = [246, 161, 160, 159, 158, 157, 173, 7, 163, 144, 145, 153, 154, 155, 33, 133]

def mediapipe_process(rgb):
    rgb.flags.writeable = False
    results = FACE_MESH.process(rgb)

    rgb.flags.writeable = True
    if results.multi_face_landmarks:
        print(f'LEN:{len(results.multi_face_landmarks)}')
        if len(results.multi_face_landmarks) is None:
            return -1, -1
        
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

        return ((left_ear + right_ear) / 2), int((cx_max - cx_min)*(cy_max - cy_min))
    else:
        return -1, -1