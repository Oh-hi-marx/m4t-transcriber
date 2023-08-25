import torch
from seamless_communication.models.inference import Translator
import torchaudio

import time 

def resample(path, outpath):
    resample_rate = 16000
    waveform, sample_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    torchaudio.save(outpath, resampled_waveform, resample_rate)

path_input = 'saycut.wav'
outpath = 'resmapled.wav'
resample(path_input, outpath)

# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_large", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"), dtype=torch.float16)
st = time.time()
for i in range(10):

    translated_text, _, _ = translator.predict(outpath, "s2tt", 'eng')
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
print(translated_text)