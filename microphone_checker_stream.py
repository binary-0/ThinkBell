import keyboard
import pyaudio
import wave
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
import math

def process_stop():
    global isStop
    isStop = 1

def mfcc_process():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    #print("* recording")

    frames = []

    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #     data = stream.read(CHUNK)
    #     frames.append(data)
    stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)


    global isStop
    isStop = 0
    while isStop == 0:
        data = stream.read(CHUNK)
        frames.append(data)
        #if keyboard.is_pressed('s'):
            # print("\n*Done Recording")
            #break


    #print("\n* done recording")
    # print(type(data))


    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    path = 'output.wav'
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
    silence = np.count_nonzero(abs(delta2_mfcc) < 1)
    size = delta2_mfcc.size

    print("\ntime of silence : ", silence)
    print("total : ", size)
    print("speaking rate : ", 100 - silence / size * 100, "%")

    return silence, size

    '''
    plt.figure(figsize=(12, 1))
    librosa.display.specshow(delta2_mfcc)
    # plt.ylabel('MFCC coeffs')
    plt.xlabel('Time')
    plt.title('MFCC')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    '''
