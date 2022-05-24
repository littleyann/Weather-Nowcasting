import matplotlib.pyplot as plt
import librosa.core as lc
import librosa
import numpy as np
import librosa.display
import torch
from conv_stft import STFT


def read_file(path, f=16000):
    wav, _ = lc.load(path, f)
    wav_tensor = torch.FloatTensor(wav)
    wav_tensor = wav_tensor.unsqueeze(0)
    return wav, wav_tensor


def spectrogram(wav, window='hann', fft_len=512, win_hop=160, win_len=320):
    stft = STFT(fft_len=fft_len, win_hop=win_hop, win_len=win_len, win_type=window)
    real, image = stft.transform(wav, return_type='realimag')
    real = real.squeeze(0).numpy()
    image = image.squeeze(0).numpy()
    spec = real + 1j * image
    spec_db = librosa.amplitude_to_db(abs(spec), ref=np.max)
    return spec_db


def plot():
    clean_path = 'clean.wav'
    noisy_path = 'noisy.wav'
    enhance_path = 'enhance.wav'

    clean_wav, clean_wav_tensor = read_file(clean_path)
    noisy_wav, noisy_wav_tensor = read_file(noisy_path)
    enhance_wav, enhance_wav_tensor = read_file(enhance_path)

    clean_spec = spectrogram(clean_wav_tensor)
    noisy_spec = spectrogram(noisy_wav_tensor)
    enhance_spec = spectrogram(enhance_wav_tensor)

    fig = plt.figure(figsize=(25, 6))
    grid = plt.GridSpec(3, 3, wspace=0.5, hspace=0.5)
    plt.subplot(grid[0, 0])
    plt.plot(clean_wav)
    plt.xticks([])

    plt.subplot(grid[0, 1])
    plt.plot(noisy_wav)
    plt.xticks([])

    plt.subplot(grid[0, 2])
    plt.plot(enhance_wav)
    plt.xticks([])

    plt.subplot(grid[1:2, 0])
    librosa.display.specshow(clean_spec, sr=16000, hop_length=160, x_axis='time', y_axis='hz')

    plt.subplot(grid[1:2, 1])
    librosa.display.specshow(noisy_spec, sr=16000, hop_length=160, x_axis='time', y_axis='hz')

    plt.subplot(grid[1:2, 2])
    librosa.display.specshow(enhance_spec, sr=16000, hop_length=160, x_axis='time', y_axis='hz')


if __name__ == '__main__':
    plot()


