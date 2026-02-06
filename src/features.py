"""Adapted from https://github.com/nesl/asvspoof2019/blob/master/model_main.py (MIT license)"""
import librosa
import numpy as np
import torchvision.transforms as T


def get_log_spectrum(x):
    s = librosa.stft(x, n_fft=2048, win_length=2048, hop_length=512)
    a = np.abs(s)**2
    feat = librosa.power_to_db(a)
    return feat

log_spectrum_extractor = T.Compose([
    lambda x: librosa.util.normalize(x),
    lambda x: get_log_spectrum(x),
    T.ToTensor()
])