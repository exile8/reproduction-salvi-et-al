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

def isolate_frequency_band(spec, num_bands, isolated_band):
    masked = spec.clone()
    
    freq_bins = spec.shape[-2]
    band_size = freq_bins // num_bands

    for i in range(num_bands):
        if i != isolated_band:
            start = i * band_size
            end = (i + 1) * band_size if i < num_bands - 1 else freq_bins
            masked[..., start:end, :] = 0.

    return masked

def band_isolator(num_bands, isolated_band):
    return lambda x: isolate_frequency_band(x, num_bands, isolated_band)