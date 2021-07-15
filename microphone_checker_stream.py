# import keyboard
import pyaudio
import wave
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
import math
import moviepy.editor as mp

sr1=0
sr2=0
sr3=0
sr4=0

sc1=0
sc2=0
sc3=0
sc4=0

x1=0
x2=0
x3=0
x4=0

def process_stop():
    global isStop
    isStop = 1

def mfcc_process(audio_name,temp_name,cut_point):
    # mp.VideoFileClip(video_name).subclip(0, cut_point).audio.write_audiofile(temp_name)
    mp.AudioFileClip(audio_name).subclip(0,cut_point).write_audiofile(temp_name)
    #지금 여기선 video_name자리에 SampleVideo1.mp4를 집어 넣어도 무방
    path = temp_name
            # path = 'sample.wav' #파일 업로드 시 사용
    sample_rate = 16000

    global x1
    global x2
    global x3
    global x4

    x = librosa.load(path, sample_rate)[0]
    # plt.figure(figsize=(9, 3))
    # librosa.display.waveplot(x) 
    # plt.savefig('result1.png',bbox_inches='tight')

    if audio_name == "SampleAudio1.wav":
        x1=x
    elif audio_name == "SampleAudio2.wav":
        x2=x
    elif audio_name == "SampleAudio3.wav":
        x3=x
    elif audio_name == "SampleAudio4.wav":
        x4=x

    S = librosa.feature.melspectrogram(x, sr=sample_rate, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=1)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    # print(delta2_mfcc)

    global sr1
    global sr2
    global sr3
    global sr4
    global sc1
    global sc2
    global sc3
    global sc4
    # print(x)
    # print(S)
    # print(delta2_mfcc)
    silence = np.count_nonzero(abs(delta2_mfcc) < 2)
    size = delta2_mfcc.size
    SpeakingRate = int(100 - silence / size * 100)

    # print(np.count_nonzero(delta2_mfcc))

    # print("\ntime of silence : ", silence)
    # print("total : ", size)
    # print("speaking rate : ", SpeakingRate, "%")

    speakCount =0
    speakIndex = np.where(abs(delta2_mfcc)>5)[1]
    # print(speakIndex.size)

    toggle = True
    for i in speakIndex:
        if i-3&i-2&i-1&i&i+1&i+2&i+3 in speakIndex :
            if Toggle == True :
                speakCount+=1
                # print(i)
                Toggle = False
        else :
            Toggle = True   

    print("[{}] 발화율 : {}%, 발화횟수 : {}회".format(audio_name, SpeakingRate, speakCount))

    if audio_name == "SampleAudio1.wav":
        sr1 = SpeakingRate
        sc1 = speakCount
    elif audio_name == "SampleAudio2.wav":
        sr2 = SpeakingRate
        sc2 = speakCount
    elif audio_name == "SampleAudio3.wav":
        sr3 = SpeakingRate
        sc3 = speakCount
    elif audio_name == "SampleAudio4.wav":
        sr4 = SpeakingRate
        sc4 = speakCount
         
    

    # plt.figure(figsize=(9, 3))
    # librosa.display.waveplot(x) 
    # plt.savefig('result.png',bbox_inches='tight')
    # plt.show()
    # return SpeakingRate

def getAD1():
    return sr1, sc1
def getAD2():
    return sr2, sc2
def getAD3():
    return sr3, sc3
def getAD4():
    return sr4, sc4

def plot1():
    plt.figure(figsize=(9, 3))
    librosa.display.waveplot(x1)
    plt.savefig('mfccResult1.png',bbox_inches='tight')
def plot2():
    plt.figure(figsize=(9, 3))
    librosa.display.waveplot(x2)
    plt.savefig('mfccResult2.png',bbox_inches='tight')
def plot3():
    plt.figure(figsize=(9, 3))
    librosa.display.waveplot(x3)
    plt.savefig('mfccResult3.png',bbox_inches='tight')
def plot4():
    plt.figure(figsize=(9, 3))
    librosa.display.waveplot(x4)
    plt.savefig('mfccResult4.png',bbox_inches='tight')