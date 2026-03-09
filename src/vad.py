import webrtcvad
import numpy as np
import torch

def trim_silence(audio, sample_rate=16000, 
                 aggressiveness=3, frame_duration_ms=30):

    if frame_duration_ms not in (10, 20, 30):
        raise ValueError("frame_duration_ms must be 10, 20, or 30")
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError("sample_rate must be 8000, 16000, 32000, or 48000")

    vad = webrtcvad.Vad(aggressiveness)

    audio_int16 = (audio * 32767).astype(np.int16)

    frame_length = int(sample_rate * frame_duration_ms / 1000)
    num_frames = len(audio_int16) // frame_length

    voiced_audio = []

    for i in range(num_frames):
        start = i * frame_length
        frame_data = audio_int16[start:start + frame_length]

        if len(frame_data) != frame_length:
            continue

        frame = frame_data.tobytes()

        try:
            if vad.is_speech(frame, sample_rate):
                voiced_audio.append(audio[start:start + frame_length])
        except Exception as e:
            print("VAD error:", e)

    if voiced_audio:
        return np.concatenate(voiced_audio)
    else:
        return audio


def vad_mask(audio, sample_rate=16000, 
             aggressiveness=3, frame_duration_ms=30,
             min_consecutive=3):
    
    if frame_duration_ms not in (10, 20, 30):
        raise ValueError("frame_duration_ms must be 10, 20, or 30")
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError("sample_rate must be 8000, 16000, 32000, or 48000")
    
    vad = webrtcvad.Vad(aggressiveness)
    
    audio_int16 = (audio * 32767).astype(np.int16)
    
    frame_length = int(sample_rate * frame_duration_ms / 1000)
    total_samples = len(audio_int16)
    num_frames = total_samples // frame_length
    
    remainder = total_samples % frame_length
    if remainder > 0:
        padding = np.zeros(frame_length - remainder, dtype=np.int16)
        audio_padded = np.concatenate([audio_int16, padding])
        num_frames += 1
    else:
        audio_padded = audio_int16
    
    is_speech = []
    for i in range(num_frames):
        start = i * frame_length
        frame_data = audio_padded[start:start + frame_length]
        try:
            is_speech.append(vad.is_speech(frame_data.tobytes(), sample_rate))
        except Exception:
            is_speech.append(False)
    
    first_speech_frame = None
    consecutive = 0
    for i in range(num_frames):
        if is_speech[i]:
            consecutive += 1
            if consecutive >= min_consecutive:
                first_speech_frame = i - min_consecutive + 1
                break
        else:
            consecutive = 0
    
    last_speech_frame = None
    consecutive = 0
    for i in range(num_frames - 1, -1, -1):
        if is_speech[i]:
            consecutive += 1
            if consecutive >= min_consecutive:
                last_speech_frame = i + min_consecutive - 1
                break
        else:
            consecutive = 0
    
    if first_speech_frame is None or last_speech_frame is None:
        return np.ones(total_samples, dtype=np.float32)
    
    mask = np.ones(total_samples, dtype=np.float32)
    speech_start = first_speech_frame * frame_length
    speech_end = min((last_speech_frame + 1) * frame_length, total_samples)
    mask[speech_start:speech_end] = 0.0
    
    return mask