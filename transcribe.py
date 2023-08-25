"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""
import torchaudio
import pyaudio
import wave
import os 
import shutil
import torch
from seamless_communication.models.inference import Translator




class M4t:
    def __init__(self):
        self.translator = Translator("seamlessM4T_large", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"), dtype=torch.float16)

    def translate(self, path):
        
        if(os.path.exists('segments')):
            shutil.rmtree('segments')
        os.makedirs("segments", exist_ok=True)
        os.system('ffmpeg -i "' + path + '" -f segment -segment_time 28 -c copy -acodec pcm_s16le -ac 1 -ar 16000 segments/out%03d.wav' )
        
        segments= os.listdir('segments')
        segments.sort()
        transcriptions = []

        for segment in segments:
            translated_text, _, _ = self.translator.predict('segments' + os.sep + segment, "s2tt", 'eng')
            transcriptions.append(str(translated_text))

        translation = " ".join(transcriptions)
        shutil.rmtree('segments')
        return translation

if __name__ == "__main__":
    path = 'lexcut.wav'
    m4t = M4t()
    translation = m4t.translate(path)
    print(translation)  