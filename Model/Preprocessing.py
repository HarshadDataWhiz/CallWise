import pandas as pd
import librosa
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
#read_audio from silero_vad is not use librosa is used because with mono = False, we can directly separate lc, rc
import torch
import torchaudio.functional as F

VAD_MODEL = load_silero_vad()



class Preprocessing:
    def __init__(self, audio_path, sr=None):
        self.audio_path = audio_path
        self.sr = sr

    def separate_channels(self):
        audio, sr = librosa.load(self.audio_path, sr=self.sr, mono=False)
        self.lc, self.rc = audio[0], audio[1]
        self.sr = sr
        return

    def voice_activity_detection(self, audio):
        
        audio_input = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        speech_timestamps = get_speech_timestamps(
            audio_input, VAD_MODEL, sampling_rate=self.sr,
            threshold=0.55,             # moderate sensitivity
            min_speech_duration_ms=250, # ignore very short noises
            min_silence_duration_ms=400 # split segments at natural pauses
        )
        return speech_timestamps
    
    def extract_segments(self):
        left_speech_timestamps = self.voice_activity_detection(self.lc)
        right_speech_timestamps = self.voice_activity_detection(self.rc)
        self.lc = [self.lc[t['start']:t['end']] for t in left_speech_timestamps]
        self.rc = [self.rc[t['start']:t['end']] for t in right_speech_timestamps]

        self.lc = [F.preemphasis(torch.from_numpy(seg).float()) for seg in self.lc]
        self.rc = [F.preemphasis(torch.from_numpy(seg).float()) for seg in self.rc]

        self.lc = torch.cat(self.lc, dim=-1).numpy()
        self.rc = torch.cat(self.rc, dim=-1).numpy()
        return self.lc , self.rc, self.sr
