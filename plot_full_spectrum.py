import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def plot_peaks(spectrum:np.array):
    peaks_idx, _ = find_peaks(spectrum[1,:],height=4, distance=2000)
    frequencies = spectrum[0,peaks_idx]
    peaks = spectrum[1,peaks_idx]
    plt.plot(frequencies, peaks, "x")
    # Print peaks from biggest to least
    peaks_idx = np.argsort(peaks)[::-1]
    print(f"Fundamental: {(frequencies[peaks_idx])[0]}")

def plot_spectrum(spectrum:np.array):
    frequencies = spectrum[0,:]
    channel_1 = spectrum[1,:]
    plt.plot(frequencies, channel_1)

if __name__=="__main__":
    spectrum = np.load("full_spectrum.npy")
    plot_spectrum(spectrum)
    plot_peaks(spectrum)
    plt.show()