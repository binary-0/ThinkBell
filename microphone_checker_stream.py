# import keyboard
import pyaudio
import wave
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
import math
import moviepy.editor as mp

def process_stop():
    global isStop
    isStop = 1

def mfcc_process(video_name,temp_name,cut_point):
    mp.VideoFileClip(video_name).subclip(0, cut_point).audio.write_audiofile(temp_name)
    #지금 여기선 video_name자리에 SampleVideo1.mp4를 집어 넣어도 무방
    path = temp_name
            # path = 'sample.wav' #파일 업로드 시 사용
    sample_rate = 16000

    x = librosa.load(path, sample_rate)[0]
    S = librosa.feature.melspectrogram(x, sr=sample_rate, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=1)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    # print(delta2_mfcc)


    # print(x)
    # print(S)
    # print(delta2_mfcc)
    silence = np.count_nonzero(abs(delta2_mfcc) < 2)
    size = delta2_mfcc.size
    SpeakingRate = 100 - silence / size * 100

    # print(np.count_nonzero(delta2_mfcc))

    # print("\ntime of silence : ", silence)
    # print("total : ", size)
    # print("speaking rate : ", SpeakingRate, "%")

    print(video_name, SpeakingRate)

    return SpeakingRate