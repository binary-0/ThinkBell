# import keyboard
# import pyaudio
import wave
# import matplotlib.pyplot as plt
# import librosa.display
# import librosa
# import numpy as np
import math

import io
import numpy as np
import torch
torch.set_num_threads(1)
import torchaudio
import matplotlib
import matplotlib.pylab as plt
torchaudio.set_audio_backend("soundfile")
import pyaudio
import threading

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

def process_stop():
    global isStop
    isStop = 1

def mfcc_process():
    
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

    (get_speech_ts,
    get_speech_ts_adaptive,
    save_audio,
    read_audio,
    state_generator,
    single_audio_stream,
    collect_chunks) = utils

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

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    data = []
    voiced_confidences = []

    global isStop
    isStop = 0

    # pp = ProgressPlot(plot_names=["Silero VAD"], line_names=["speech probabilities"], x_label="audio chunks")


    # stop_listener = threading.Thread(target=process_stop)
    # stop_listener.start()

    while isStop==0:
        audio_chunk = stream.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0))

        # in case you want to save the audio later
        data.append(audio_chunk)

        audio_int16 = np.frombuffer(audio_chunk, np.int16)

        audio_float32 = int2float(audio_int16)

        # get the confidences and add them to the list to plot them later
        vad_outs = validate(model, torch.from_numpy(audio_float32))

        # get the confidence value so that jupyterplot can process it
        new_confidence = vad_outs[:, 1].numpy()[0].item()
        # new_confidence = vad_outs[:, 1]

        voiced_confidences.append(new_confidence)

        print("heyheyhey")

        # pp.update(new_confidence)
        plt.plot(voiced_confidences)
        plt.pause(0.0000001)
        


    # pp.finalize()
    # plt.plot(new_confidence)
    # plt.figure(figsize=(12, 6))
        count = sum(map(lambda x: x > 0.7, voiced_confidences))
        length = len(voiced_confidences)
        print("total length : ", length)
        print("count speak : ", count)
    # print("speaking rate : ", (count/length)*100, "%")
        plt.savefig('result.png', bbox_inches='tight')
    # plt.show()

        if count<1:
            count=1
        if length<1:
            length=2

        silence = length-count
        size = length

    silence = length-count
    size = length

    return silence, size