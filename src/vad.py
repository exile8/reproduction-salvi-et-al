import webrtcvad
import numpy as np

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
