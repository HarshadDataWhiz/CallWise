import pandas as pd
import librosa
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
#read_audio from silero_vad is not use librosa is used because with mono = False, we can directly separate lc, rc
import torch
import torchaudio.functional as F
import ast
import numpy as np 
from datasets import Dataset
VAD_MODEL = load_silero_vad()


class inference:
    def __init__(self, audio_path ,asr_pipeline ,batchsize, sr=16000):
        self.audio_path = audio_path
        self.sr = sr
        self.asr_pipeline = asr_pipeline
        audio, _ = librosa.load(self.audio_path, sr=self.sr, mono=False)
        self.lc, self.rc = audio[0], audio[1]

        self.batchsize = batchsize
        return

    def Transcribe(self):
        # -----------------------------
        # 1. VAD for both channels
        # -----------------------------
        speech_timestamps_lc = get_speech_timestamps(
            self.lc, VAD_MODEL,
            sampling_rate=self.sr,
            threshold=0.55,
            min_speech_duration_ms=500,
            min_silence_duration_ms=1100
        )

        speech_timestamps_rc = get_speech_timestamps(
            self.rc, VAD_MODEL,
            sampling_rate=self.sr,
            threshold=0.55,
            min_speech_duration_ms=500,
            min_silence_duration_ms=1100
        )

        # -----------------------------
        # 2. Collect all segments
        # -----------------------------
        segments = []

        for ts in speech_timestamps_lc:
            segments.append({
                "audio": self.lc[ts["start"]:ts["end"]],
                "sampling_rate": self.sr,
                "speaker": "Agent",
                "start": ts["start"]
            })

        for ts in speech_timestamps_rc:
            segments.append({
                "audio": self.rc[ts["start"]:ts["end"]],
                "sampling_rate": self.sr,
                "speaker": "Customer",
                "start": ts["start"]
            })

        # -----------------------------
        # 3. Sort by time (important!)
        # -----------------------------
        segments = sorted(segments, key=lambda x: x["start"])

        # -----------------------------
        # 4. Batched ASR inference
        # -----------------------------
        # Extract audio arrays and sampling rates into separate lists for pipeline input
        audio_inputs = []
        speaker_labels = []
        for s in segments:
            audio_inputs.append({
                "array": s["audio"],
                "sampling_rate": s["sampling_rate"]
            })
            speaker_labels.append(s["speaker"])

        outputs = self.asr_pipeline(
            audio_inputs,
            batch_size=self.batchsize,
            return_timestamps=False
        )

        # -----------------------------
        # 5. Reconstruct conversation
        # -----------------------------
        res = ""
        for out, speaker in zip(outputs, speaker_labels):
            res += f"\n {speaker} : {out['text']}"

        return res