import pandas as pd
import librosa
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
#read_audio from silero_vad is not use librosa is used because with mono = False, we can directly separate lc, rc
import torch
import torchaudio.functional as F
import ast
import numpy as np 
VAD_MODEL = load_silero_vad()



class Preprocessing:
    def __init__(self, audio_path, trans ,sr=None):
        self.audio_path = audio_path
        self.sr = sr

        # Convert string repr of lists into real Python lists
        start_times = ast.literal_eval(trans['start_time'])
        end_times = ast.literal_eval(trans['end_time'])
        speakers = ast.literal_eval(trans['speaker'])
        transcripts = ast.literal_eval(trans['transcription'])

        # Initialize
        agent_time, agent_trans = [], []
        customer_time, customer_trans = [], []

        # Iterate over all segments
        for s, e, spk, txt in zip(start_times, end_times, speakers, transcripts):
            if spk.lower() == "agent":
                agent_time.append((s, e))
                agent_trans.append(txt)
            elif spk.lower() == "customer":
                customer_time.append((s, e))
                customer_trans.append(txt)
        
        self.agent_time = agent_time
        self.agent_trans = agent_trans
        self.customer_time = customer_time
        self.customer_trans = customer_trans

        return

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
        audio = [audio[t['start']:t['end']] for t in speech_timestamps]
        audio = [F.preemphasis(torch.from_numpy(seg).float()) for seg in audio]
        # print(audio)
        # print(type(audio))
        if len(audio) != 0:
            audio = torch.cat(audio, dim=-1).numpy()
            return audio
        else:
            return np.array([])
