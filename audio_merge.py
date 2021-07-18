from pydub import AudioSegment
from pydub.playback import play

audio1 = AudioSegment.from_file("SampleAudio1.wav") #your first audio file
audio2 = AudioSegment.from_file("SampleAudio2.wav") #your second audio file
audio3 = AudioSegment.from_file("SampleAudio3.wav") #your third audio file
audio4 = AudioSegment.from_file("SampleAudio4.wav") #your third audio file
mixed = audio1.overlay(audio2).overlay(audio3).overlay(audio4)        #Further combine , superimpose audio files
#If you need to save mixed file
mixed.export("SampleAudioAll.wav", format='wav') #export mixed  audio file
# play(mixed1)    