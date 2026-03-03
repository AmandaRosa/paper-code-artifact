import numpy as np
from scipy.signal import butter, filtfilt


def LowPassFilter(bearing_signal, fs=50000):

    # Define filter parameters
    cutoff_freq = 2000  # Cutoff frequency for the low-pass filter (Hz)
    nyquist_freq = 0.5 * fs  # Nyquist frequency
    normalized_cutoff_freq = cutoff_freq / nyquist_freq  # Normalize cutoff frequency

    # Design a low-pass Butterworth filter
    order = 4  # Filter order
    b, a = butter(order, normalized_cutoff_freq, btype="low")

    # Apply the filter to the raw signal
    filtered_signal = filtfilt(b, a, bearing_signal)

    return np.array(filtered_signal)