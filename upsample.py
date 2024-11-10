import librosa
import librosa.display
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def main():
    fs = 16000
    f0 = 2000
    f1 = 6000
    len = 5 * fs
    usc = 3 # upsampling constant
    n = np.arange(len)
    s = 0.9 * np.sin(2*np.pi*f0*n/fs) + 0.9 * np.sin(2*np.pi*f1*n/fs)
    tmp = np.expand_dims(s, axis=0)
    y = np.append(tmp, np.zeros([usc - 1, tmp.shape[1]]), axis=0).flatten('F')

    b, a = signal.butter(8, fs/2, btype='lowpass', fs=fs*usc)
    # s = signal.lfilter(b, a, s)
    # yp = y
    yp = signal.lfilter(b, a, y)

    n_fft = 2048
    half_n_fft = n_fft // 2
    
    win_ana = np.sqrt(np.hanning(n_fft+1))[:n_fft]
    s_stft = librosa.stft(s, n_fft=n_fft, hop_length=half_n_fft, window=win_ana)
    y_stft = librosa.stft(y, n_fft=n_fft, hop_length=half_n_fft, window=win_ana)
    yp_stft = librosa.stft(yp, n_fft=n_fft, hop_length=half_n_fft, window=win_ana)

    plt.figure(figsize=(8, 6))
    axs = plt.subplot(3, 1, 1)
    tmp = librosa.amplitude_to_db(np.abs(s_stft), ref=half_n_fft)
    librosa.display.specshow(tmp, x_axis='time', y_axis='hz', sr=fs, hop_length=half_n_fft)
    plt.colorbar()
    plt.subplot(3, 1, 2, sharex=axs)
    tmp = librosa.amplitude_to_db(np.abs(y_stft), ref=half_n_fft)
    librosa.display.specshow(tmp, x_axis='time', y_axis='hz', sr=fs * usc, hop_length=half_n_fft)
    plt.colorbar()
    plt.subplot(3, 1, 3, sharex=axs)
    tmp = librosa.amplitude_to_db(np.abs(yp_stft), ref=half_n_fft)
    librosa.display.specshow(tmp, x_axis='time', y_axis='hz', sr=fs * usc, hop_length=half_n_fft)
    plt.colorbar()
    plt.tight_layout()
    plt.figure(figsize=(8, 6))

    axs = plt.subplot(3, 1, 1)
    f_bin = np.arange(0, half_n_fft + 1) * fs / n_fft
    pow = np.mean(np.abs(s_stft)**2, axis=1)
    plt.plot(f_bin, pow)
    plt.subplot(3, 1, 2, sharex=axs)
    f_bin = np.arange(0, half_n_fft + 1) * (fs * usc) / n_fft
    pow = np.mean(np.abs(y_stft)**2, axis=1)
    plt.plot(f_bin, pow)
    plt.subplot(3, 1, 3, sharex=axs)
    pow = np.mean(np.abs(yp_stft)**2, axis=1)
    plt.plot(f_bin, pow)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()