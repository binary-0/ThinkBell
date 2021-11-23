# import io
# import numpy as np
# import torch
# torch.set_num_threads(1)
# import torchaudio
# import matplotlib
# import matplotlib.pylab as plt
# torchaudio.set_audio_backend("soundfile")
# import pyaudio
# import time

# plt.rcParams["figure.figsize"]=(12,3)

# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#                               model='silero_vad')

# global liveSC
# liveSC = 0

# (get_speech_ts,
#  get_speech_ts_adaptive,
#  save_audio,
#  read_audio,
#  state_generator,
#  single_audio_stream,
#  collect_chunks) = utils

# # Taken from utils_vad.py
# def validate(model,
#              inputs: torch.Tensor):
#     with torch.no_grad():
#         outs = model(inputs)
#     return outs

# # Provided by Alexander Veysov
# def int2float(sound):
#     abs_max = np.abs(sound).max()
#     sound = sound.astype('float32')
#     if abs_max > 0:
#         sound *= 1/abs_max
#     sound = sound.squeeze()  # depends on the use case
#     return sound

# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# SAMPLE_RATE = 16000
# CHUNK = int(SAMPLE_RATE / 10)

# audio = pyaudio.PyAudio()

# frames_to_record = 20 # frames_to_record * frame_duration_ms = recording duration
# frame_duration_ms = 250

# stream = audio.open(format=FORMAT,
#                     channels=CHANNELS,
#                     rate=SAMPLE_RATE,
#                     input=True,
#                     frames_per_buffer=CHUNK)
# data = []
# voiced_confidences = []
# test_confidences = []

# # from jupyterplot import ProgressPlot

# continue_recording = True

# def stop():
#     input("Press Enter to stop the recording:")
#     global continue_recording
#     continue_recording = False

# def start_recording():
#     stream = audio.open(format=FORMAT,
#                         channels=CHANNELS,
#                         rate=SAMPLE_RATE,
#                         input=True,
#                         frames_per_buffer=CHUNK)
#     startTime=time.time()
#     data = []
#     voiced_confidences = []
#     test_confidences = []

#     global continue_recording
#     continue_recording = True

#     # pp = ProgressPlot(plot_names=["Silero VAD"], line_names=["speech probabilities"], x_label="audio chunks")

#     isAgain = False
#     temp_confidence = []
#     speechCount = 0
#     checkTime = 0
#     while True:
#         # print("hehehe")
#         audio_chunk = stream.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0))

#         # in case you want to save the audio later
#         data.append(audio_chunk)

#         audio_int16 = np.frombuffer(audio_chunk, np.int16);

#         audio_float32 = int2float(audio_int16)

#         # get the confidences and add them to the list to plot them later
#         vad_outs = validate(model, torch.from_numpy(audio_float32))

#         # get the confidence value so that jupyterplot can process it
#         new_confidence = vad_outs[:, 1].numpy()[0].item()
#         # new_confidence = vad_outs[:, 1]

#         #잘 되는 코드 하나 근데 쪼금 불안정
#         # if new_confidence>0.7 and isAgain is False : #처음 시작 타임체크
#         #     checkTime = time.time()
#         #     isAgain = True
#         # elif new_confidence>0.7 and isAgain is True and Toggle is False: #다음 일때, 연속 체크
#         #     nowTime = time.time()
#         #     temp_confidence.append(new_confidence)
#         #     temp_avg = sum(temp_confidence)/len(temp_confidence)
#         #     if nowTime - checkTime > 7 and temp_avg>0.5: #7초가 지났을 때
#         #         speechCount+=1
#         #         isAgain=False
#         #         Toggle=True
#         #         print("발표!", nowTime-startTime)
#         #         temp_confidence.clear()
#         # if new_confidence<=0.7:
#         #     Toggle=False

#         if new_confidence>0.7 and isAgain is False:
#             isAgain = True
#             checkTime = time.time()

#         if isAgain is True:
#             temp_confidence.append(new_confidence)
#             nowTime=time.time()
#             if nowTime - checkTime > 6: #6초의 타임스팬에서
#                 temp_avg = sum(temp_confidence)/len(temp_confidence)
#                 temp_spoken = sum(map(lambda x: x > 0.6, temp_confidence))
#                 temp_spoken_ratio = temp_spoken/len(temp_confidence)
#                 if temp_spoken_ratio>0.4: #말을 한 비율이 40%정도면 발표로 인식
#                     speechCount+=1
#                     balpyo_time = nowTime-startTime
#                     print("발표! {}분 {}초".format(int(balpyo_time/60), int(balpyo_time%60)))
#                 temp_confidence.clear()
#                 isAgain=False


#         if len(voiced_confidences)>50 :
#             del voiced_confidences[0]
#         voiced_confidences.append(new_confidence)
#         test_confidences.append(new_confidence)
        
#         # micSC.value = speechCount

#         # print(type(voiced_confidences))

#         # pp.update(new_confidence)
#         # plt.clf()
#         # plt.ylim([0,1])
#         # plt.xticks([])
#         # plt.axhline(y=0.7)
#         # plt.plot(voiced_confidences)
#         # plt.pause(0.00001)

#     print("\n\n총 발표 횟수 : ",speechCount)
#     # pp.finalize()
#     # plt.plot(new_confidence)
#     # plt.figure(figsize=(12, 6))
#     endTime = time.time()
#     timeSpan = endTime-startTime
#     # print(timeSpan)
#     # print(voiced_confidences)
#     count = sum(map(lambda x: x > 0.7, test_confidences))
#     length = len(test_confidences)

#     # print("발화 비율 : ", (count/length)*100, "%")
#     # plt.savefig('vad_result.png', bbox_inches='tight')
#     # plt.show()


# # print(type(voiced_confidences))
# # start_recording()
# # count = sum(map(lambda x : x<0.2, voiced_confidences))
# # print("total length", len(voiced_confidences))
# # print('Count of odd numbers in a list : ', count)

# # def returnLiveSC():
# #     global liveSC
# #     return liveSC



import io
import numpy as np
import torch
torch.set_num_threads(1)
import torchaudio
import matplotlib
import matplotlib.pylab as plt
torchaudio.set_audio_backend("soundfile")
import pyaudio
import time

plt.rcParams["figure.figsize"]=(12,3)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad')

global liveSC
liveSC = 0

(get_speech_ts,
 get_speech_ts_adaptive,
 save_audio,
 read_audio,
 state_generator,
 single_audio_stream,
 collect_chunks) = utils

# Taken from utils_vad.py
def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze()  # depends on the use case
    return sound

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)

audio = pyaudio.PyAudio()

frames_to_record = 20 # frames_to_record * frame_duration_ms = recording duration
frame_duration_ms = 250

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
data = []
voiced_confidences = []
test_confidences = []

# from jupyterplot import ProgressPlot
import threading

continue_recording = True

def stop():
    input("Press Enter to stop the recording:")
    global continue_recording
    continue_recording = False

def start_recording(micSC):
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    startTime=time.time()
    data = []
    voiced_confidences = []
    test_confidences = []

    global continue_recording
    continue_recording = True

    # pp = ProgressPlot(plot_names=["Silero VAD"], line_names=["speech probabilities"], x_label="audio chunks")

    isAgain = False
    temp_confidence = []
    speechCount = 0
    checkTime = 0
    while True:
        audio_chunk = stream.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0))

        # in case you want to save the audio later
        data.append(audio_chunk)

        audio_int16 = np.frombuffer(audio_chunk, np.int16);

        audio_float32 = int2float(audio_int16)

        # get the confidences and add them to the list to plot them later
        vad_outs = validate(model, torch.from_numpy(audio_float32))

        # get the confidence value so that jupyterplot can process it
        new_confidence = vad_outs[:, 1].numpy()[0].item()
        # new_confidence = vad_outs[:, 1]

        #잘 되는 코드 하나 근데 쪼금 불안정
        # if new_confidence>0.7 and isAgain is False : #처음 시작 타임체크
        #     checkTime = time.time()
        #     isAgain = True
        # elif new_confidence>0.7 and isAgain is True and Toggle is False: #다음 일때, 연속 체크
        #     nowTime = time.time()
        #     temp_confidence.append(new_confidence)
        #     temp_avg = sum(temp_confidence)/len(temp_confidence)
        #     if nowTime - checkTime > 7 and temp_avg>0.5: #7초가 지났을 때
        #         speechCount+=1
        #         isAgain=False
        #         Toggle=True
        #         print("발표!", nowTime-startTime)
        #         temp_confidence.clear()
        # if new_confidence<=0.7:
        #     Toggle=False

        if new_confidence>0.7 and isAgain is False:
            isAgain = True
            checkTime = time.time()

        if isAgain is True:
            temp_confidence.append(new_confidence)
            nowTime=time.time()
            if nowTime - checkTime > 6: #6초의 타임스팬에서
                temp_avg = sum(temp_confidence)/len(temp_confidence)
                temp_spoken = sum(map(lambda x: x > 0.6, temp_confidence))
                temp_spoken_ratio = temp_spoken/len(temp_confidence)
                if temp_spoken_ratio>0.4: #말을 한 비율이 40%정도면 발표로 인식
                    speechCount+=1
                    balpyo_time = nowTime-startTime
                    print("발표! {}분 {}초".format(int(balpyo_time/60), int(balpyo_time%60)))
                temp_confidence.clear()
                isAgain=False


        if len(voiced_confidences)>50 :
            del voiced_confidences[0]
        voiced_confidences.append(new_confidence)
        test_confidences.append(new_confidence)
        
        micSC.value = speechCount

        # print(type(voiced_confidences))

        # pp.update(new_confidence)
        # plt.clf()
        # plt.ylim([0,1])
        # plt.xticks([])
        # plt.axhline(y=0.7)
        # plt.plot(voiced_confidences)
        # plt.pause(0.00001)

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


# print(type(voiced_confidences))
# start_recording()
# count = sum(map(lambda x : x<0.2, voiced_confidences))
# print("total length", len(voiced_confidences))
# print('Count of odd numbers in a list : ', count)

def returnLiveSC():
    global liveSC
    return liveSC
