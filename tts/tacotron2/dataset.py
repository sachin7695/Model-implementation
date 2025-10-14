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


    def mel_to_audio(self,mel, do_denorm=False, griffin_lim_iters=60):

        #griffin lim works raw spectrogram and we are in mel spectrogram
        #convert mel scale to raw spectrogram

        if do_denorm:
            mel = denormalize(mel)
        mel = db_to_amp(mel)
        spectrogram = torch.matmul(self.mel_to_spec.to(mel.device), mel).cpu().numpy() 
        audio = librosa.griffinlim(S = spectrogram,
                                   n_iter=griffin_lim_iters,
                                   hop_length=self.hop_size,
                                   win_length=self.window_size,
                                   n_fft=self.n_fft,
                                   window="hann")
        audio = (audio*32767) / max(0.01, np.max(np.abs(audio))) #int16 audio 
        audio = audio.astype(np.int16)
        return audio 
    

def build_padding_mask(lengths):

    B = lengths.size(0)
    T = torch.max(lengths).item()

    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, lengths[i]:] = 1

    return mask.bool()   


class TTSDataset(Dataset):
    def __init__(self,
                path_to_metadata,
                sample_rate=22050,
                n_fft=1024, 
                window_size=1024, 
                hop_size=256, 
                fmin=0,
                fmax=8000, 
                num_mels=80, 
                center=False, 
                normalized=False, 
                min_db=-100, 
                max_scaled_abs=4
                 ):
        super(TTSDataset, self).__init__()
        self.metadata = pd.read_csv(path_to_metadata)
        self.sample_rate = sample_rate 
        self.n_fft = n_fft 
        self.win_size = window_size 
        self.hop_size = hop_size 
        self.fmin = fmin 
        self.fmax = fmax 
        self.num_mels = num_mels 
        self.center = center 
        self.normalized = normalized 
        self.min_db = min_db 
        self.max_scaled_abs = max_scaled_abs 

        #length of each text input after tokenization
        self.transcript_lengths = []
        for text in self.metadata["normalized_transcript"]:
            self.transcript_lengths.append(len(Tokenizer().encode(text=text)))

        self.audio_proc = AudioMelConversions(
            num_mels=self.num_mels, 
            sampling_rate=self.sample_rate, 
            n_fft=self.n_fft, 
            window_size=self.win_size, 
            hop_size=self.hop_size, 
            fmin=self.fmin, 
            fmax=self.fmax, 
            center=self.center,
            min_db=self.min_db, 
            max_scaled_abs=self.max_scaled_abs
        )

    def __len__(self):
        return len(self.metadata) #length of the csv files how many lines 
    
    def __getitem__(self, idx):

        sample = self.metadata.iloc[idx] #one specific row of csv 
        path_to_audio = sample["file_path"]
        transcript = sample["normalized_transcript"]

        audio = load_wav(path_to_audio)
        mel = self.audio_proc.audio_to_mel(audio=audio, do_norm=True)
        return transcript, mel.squeeze(0) #batch dimension squeeze left [mel_bins, n_frames]
    

def TTSColator():
    tokenizer = Tokenizer()

    def _collate_fn(batch):
        texts, mels = [], []
        for b in batch:
            texts.append(tokenizer.encode(b[0])) #__getitem__ gives us transcript, melspec 
        for b in batch:
            mels.append(b[1]) 
        
        # text (B, indices list)
        ### Get Lengths of Texts and Mels ###
        input_lengths = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
        output_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)

        ### Sort by Text Length (as we will be using packed tensors later) ### 
        #torch.sort() -> gives us the sorted tensor and indices after sorting across that dim 
        #default dim = -1 

        input_lengths, sorted_idx = input_lengths.sort(descending=True)
        texts = [texts[i] for i in sorted_idx]
        mels = [mels[i] for i in sorted_idx]
        output_lengths = output_lengths[sorted_idx]

        ### Pad Text ###
        text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)


        ### Pad Mel Sequences ###
        max_target_len = max(output_lengths).item()
        num_mels = mels[0].shape[0]
        
        ### Get gate which tells when to stop decoding. 0 is keep decoding, 1 is stop ###
        mel_padded = torch.zeros((len(mels), num_mels, max_target_len))
        gate_padded = torch.zeros((len(mels), max_target_len))

        for i, mel in enumerate(mels):
            t = mel.shape[1]
            mel_padded[i, :, :t] = mel
            gate_padded[i, t-1:] = 1 #to stop the model to generate "stop generating here"
        
        mel_padded = mel_padded.transpose(1,2) #B, mel_frames, mel_bins

        return text_padded, input_lengths, mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths)
    return _collate_fn

class BatchSampler:
    def __init__(self, dataset, batch_size, drop_last=False):
        self.sampler = torch.utils.data.SequentialSampler(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_batches = self._make_batches()

    def _make_batches(self):

        indices = [i for i in self.sampler]

        if self.drop_last:

            total_size = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:total_size]

        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        random_indices = torch.randperm(len(batches))
        return [batches[i] for i in random_indices]
    
    def __iter__(self):
        for batch in self.random_batches:
            yield batch

    def __len__(self):
        return len(self.random_batches)



    
if __name__ == "__main__":
    path_to_aud = "/home/cmi_10101/Documents/coding/pytorch/" + \
    "architecture-implementation/tts/LJSpeech-1.1/wavs/LJ001-0001.wav"
    audio = load_wav(path_to_audio=path_to_aud)
    amc = AudioMelConversions() 
    amc.audio_to_mel(audio=audio, do_norm=True)