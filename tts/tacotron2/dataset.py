import pandas as pd 
import torch 
import torch.nn.functional as F 
import torchaudio 

from torch.utils.data import Dataset 
import librosa 
from tokenizer import Tokenizer 
import numpy as np  
import matplotlib.pyplot as plt 


def load_wav(path_to_audio, sr = 22050):
    audio, orig_sr = torchaudio.load(path_to_audio)
    if sr!=orig_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq = sr) 

    return audio.squeeze(0) # num_channel, num_samples 
def amp_to_db(x, min_db = -100):
    clip_val = 10**(min_db/20)
    return 20 * torch.log10(torch.clamp(x, min=clip_val)) 

def db_to_amp(x):
    return 10**(x/20) 

def normalize(x, min_db = -100, max_abs_val = 4):
    x = (x - min_db) / -min_db
    x = 2 * max_abs_val * x - max_abs_val
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)

    return x 

def denormalize(x, 
                min_db=-100, 
                max_abs_val=4):
    
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    x = (x + max_abs_val) / (2 * max_abs_val)
    x = x * -min_db + min_db

    return x

class AudioMelConversions:

    def __init__(self, 
                num_mels = 80,
                sampling_rate = 22050,
                n_fft = 1024,
                window_size = 1024,
                hop_size = 256,
                fmin = 0,
                fmax = 8000,
                center = False,
                min_db = -100,
                max_scaled_abs = 4
                ):
        self.num_mels = num_mels 
        self.sampling_rate = sampling_rate 
        self.n_fft = n_fft 
        self.window_size = window_size 
        self.hop_size = hop_size 
        self.fmin = fmin 
        self.center = center
        self.fmax = fmax 
        self.min_db = min_db 
        self.max_scaled_abs = max_scaled_abs 

        self.spec_to_mel = self.get_spec_to_mel_spec()
        self.mel_to_spec = torch.linalg.pinv(self.spec_to_mel) #moore-penrose inverse 


    
    def get_spec_to_mel_spec(self):
        mel = librosa.filters.mel(
                                sr = self.sampling_rate,
                                n_fft=self.n_fft,
                                n_mels=self.num_mels,
                                fmin=self.fmin,
                                fmax=self.fmax,
                                )
        print(mel.shape)
        print()
        print(mel)
        return torch.from_numpy(mel) 

    def audio_to_mel(self, audio, do_norm=False):

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        spectrogram = torch.stft(input=audio,
                                 n_fft=self.n_fft,
                                 hop_length=self.hop_size,
                                win_length=self.window_size,
                                window=torch.hann_window(self.window_size).to(audio.device),
                                center=self.center,
                                pad_mode="reflect",
                                normalized=False,
                                onesided=True,
                                return_complex=True)
        
        spectrogram = torch.abs(spectrogram)
        print(spectrogram.shape)
        print()
        print(spectrogram.min(), spectrogram.max())


        mel_spec = torch.matmul(self.spec_to_mel.to(spectrogram.device), spectrogram)
        print(f"min and max mel space in without normal scale {mel_spec.min()}, {mel_spec.max()}") #[7.003611244726926e-06, 4.33143949508667]

        
        mel_spec = amp_to_db(mel_spec, self.min_db) 
        if do_norm:
            mel_spec = normalize(mel_spec, min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        print(f"min and max mel space in normalized scale {mel_spec.min()}, {mel_spec.max()}") #[-4.0, 4.0]

        print(mel_spec.shape)
        print()
        print(mel_spec.min(), mel_spec.max())  
        plt.imshow(mel_spec)
        plt.show()

    
if __name__ == "__main__":
    path_to_aud = "/home/cmi_10101/Documents/coding/pytorch/" + \
    "architecture-implementation/tts/LJSpeech-1.1/wavs/LJ001-0001.wav"
    audio = load_wav(path_to_audio=path_to_aud)
    amc = AudioMelConversions() 
    amc.audio_to_mel(audio=audio, do_norm=True)