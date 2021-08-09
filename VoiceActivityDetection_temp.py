import wave
import threading
import numpy as np
import torch
torch.set_num_threads(1)
import torchaudio
import matplotlib.pylab as plt
torchaudio.set_audio_backend("soundfile")
import pyaudio
import time
from io import BytesIO
# import globalVAR

plt.rcParams["figure.figsize"]=(12,3)


model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad')

(get_speech_ts,
 get_speech_ts_adaptive,
 save_audio,
 read_audio,
 state_generator,
 single_audio_stream,
 collect_chunks) = utils

# model, utils = sendMU()
global vc1, vc2, vc3, vc4
global sc1, sc2, sc3, sc4
sc1 = 0
sc2 = 0
sc3 = 0
sc4 = 0

vc1 = []
vc2 = []
vc3 = []
vc4 = []

def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze()  # depends on the use case
    return sound

SAMPLE_RATE = 16000

frames_to_record = 20 # frames_to_record * frame_duration_ms = recording duration
frame_duration_ms = 250

# audio = pyaudio.PyAudio()

# data = []
# voiced_confidences = []
# test_confidences = []

# from jupyterplot import ProgressPlot

continue_recording = True

def stop():
    input("Press Enter to stop the recording:")
    global continue_recording
    continue_recording = False

# def multi_audio():
#     vadStart("SampleAudio1.wav")
#     vadStart("SampleAudio2.wav")
#     vadStart("SampleAudio3.wav")
#     vadStart("SampleAudio4.wav")

def vadStart(wavPATH):
    # audio = pyaudio.PyAudio()
    
    # wave.open(wavPATH, 'rb')
    with wave.open(wavPATH, 'rb') as f:
        width = f.getsampwidth()
        channels = f.getnchannels()
        rate = f.getframerate() 
        frames = f.getnframes()
    
        # stream = audio.open(
        #     format=width,
        #     channels=channels,
        #     rate=rate,
        #     frames_per_buffer=int(rate / 10),
        #     output = True
        # )
        stream = pyaudio.PyAudio().open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = 16000,
            frames_per_buffer = 1600,
            output=True
        )
        startTime=time.time()
        data = []
        voiced_confidences = [] #이게 문제가 될 수도?
        test_confidences = []

        global continue_recording
        continue_recording = True

        # stop_listener = threading.Thread(target=stop)
        # stop_listener.start()

        isAgain = False
        temp_confidence = []
        speechCount = 0
        checkTime = 0

        read = f.readframes(int(SAMPLE_RATE * frame_duration_ms / 1000.0))

    

        while read:
            stream.write(read)
            audio_chunk = stream.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0))
            # audio_chunk = f.readframes(int(rate * (frames/rate) / 1000.0))

            # audio_chunk = f.readframes(int(rate))
            
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            audio_float32 = int2float(audio_int16)

            # get the confidences and add them to the list to plot them later
            vad_outs = validate(model, torch.from_numpy(audio_float32))

            # get the confidence value so that jupyterplot can process it
            new_confidence = vad_outs[:, 1].numpy()[0].item()
            # new_confidence = vad_outs[:, 1]


            if new_confidence>0.1 and isAgain is False: #threshold 이상!
                isAgain = True
                checkTime = time.time()

            if isAgain is True:
                temp_confidence.append(new_confidence)
                nowTime=time.time()
                if nowTime - checkTime > 1: #6초의 타임스팬에서
                    temp_avg = sum(temp_confidence)/len(temp_confidence)
                    temp_spoken = sum(map(lambda x: x > 0.1, temp_confidence))
                    temp_spoken_ratio = temp_spoken/len(temp_confidence)
                    if temp_spoken_ratio>0.2: #말을 한 비율이 40%정도면 발표로 인식
                        speechCount+=1
                        balpyo_time = nowTime-startTime
                        print("발표! {}분 {}초".format(int(balpyo_time/60), int(balpyo_time%60)))
                    temp_confidence.clear()
                    isAgain=False


            if len(voiced_confidences)>50 :
                del voiced_confidences[0]
            voiced_confidences.append(new_confidence)
            test_confidences.append(new_confidence)

            # print(wavPATH,"\t\t\t", voiced_confidences, "\n\n\n")

            if wavPATH=='record1.wav':
                global vc1, sc1
                vc1 = voiced_confidences
                sc1 = speechCount
                print("나 1", sc1)
                # que.put("1")
            elif wavPATH=='record2.wav':
                global vc2, sc2
                vc2 = voiced_confidences
                sc2 = speechCount
                print("나 2", sc2)
                # que.put("2")
            elif wavPATH=='record3.wav':
                global vc3, sc3
                vc3 = voiced_confidences
                sc3 = speechCount
                print("나 3",sc3)
                # globalVAR.vc3.append(voiced_confidences)
            elif wavPATH=='record4.wav':
                global vc4, sc4
                vc4 = voiced_confidences
                sc4 = speechCount
                print("나 4",sc4)
                # globalVAR.vc4.append(voiced_confidences)
            # print(type(voiced_confidences))

            # pp.update(new_confidence)

            #여기가 플롯팅 파트인데 잠시
            # plt.switch_backend('agg')
            # plt.clf()
            # plt.ylim([0,1])
            # plt.xticks([])
            # plt.plot(voiced_confidences)
            # plt.axhline(y=0.7, color='r')
            # plt.pause(0.00001)
            # time.sleep(0.00001)
            read = f.readframes(int(SAMPLE_RATE * frame_duration_ms / 1000.0))



    print("\n\n총 발표 횟수 : ",speechCount)
    # pp.finalize()
    # plt.plot(new_confidence)
    # plt.figure(figsize=(12, 6))
    endTime = time.time()
    timeSpan = endTime-startTime
    # print(timeSpan)
    # print(voiced_confidences)
    count = sum(map(lambda x: x > 0.7, test_confidences))
    length = len(test_confidences)

    # print("발화 비율 : ", (count/length)*100, "%")
    # plt.savefig('vad_result.png', bbox_inches='tight')
    # plt.show()

def setVADdata(sampleNum):
    global vc1, vc2, vc3, vc4
    global sc1, sc2, sc3, sc4
    # return voiced_confidences
    if sampleNum==1:
        return vc1, sc1
    elif sampleNum==2:
        return vc2, sc2
    elif sampleNum==3:
        return vc3, sc3
    elif sampleNum==4:
        return vc4, sc4

def setVC1():
    global vc1
    return vc1

def setVC2():
    global vc2
    return vc2

def setSC1():
    global sc1
    return sc1

def setSC2():
    global sc2
    return sc2

def setSC3():
    global sc3
    return sc3

def setSC4():
    global sc4
    return sc4      

def getPlot(imgNum):
    plt.clf()
    plt.ylim([0,1])
    plt.xticks([])
    plt.axhline(y=0.7, color='r')
    imgNum = BytesIO()
    if imgNum==1:
        plt.plot(vc1)
        plt.savefig(imgNum, format='png', bbox_inches='tight', dpi=200)
        return imgNum
    elif imgNum==2:
        plt.plot(vc2)
        plt.savefig(imgNum, format='png', bbox_inches='tight', dpi=200)
        return imgNum
    elif imgNum==3:
        plt.plot(vc3)
        plt.savefig(imgNum, format='png', bbox_inches='tight', dpi=200)
        return imgNum
    elif imgNum==4:
        plt.plot(vc4)
        plt.savefig(imgNum, format='png', bbox_inches='tight', dpi=200)
        return imgNum
    # plt.pause(0.00001)