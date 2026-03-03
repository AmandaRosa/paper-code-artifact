import numpy as np
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.signal import hilbert
from scipy.signal import spectrogram
import matplotlib.pyplot as plt


class FrequencyDomain:

    def __init__(self, sample_length=None, pontos = 2000):
        self.sample_length = sample_length
        self.pontos = pontos

    def fft(self, signal, fs=50000, pontos=2000):
        results = []
        num_subsamples = len(signal) // self.sample_length

        for i in range(num_subsamples):
            subsample = signal[i * self.sample_length : (i + 1) * self.sample_length]
            if self.sample_length:
                subsample = subsample[: self.sample_length]  # Subsample the signal
            X_signal = np.fft.fft(subsample)
            freqs_signal = np.fft.fftfreq(len(subsample), 1 / fs)  # Frequency bins
            results.append(
                [
                    freqs_signal[: len(freqs_signal) // 2][10:pontos],
                    np.abs(X_signal)[: len(X_signal) // 2][10:pontos],
                ]
            )

        return np.array(results)

    def fft_peaks(self, signal, fs=50000, pontos=2000):
        results = []
        num_subsamples = len(signal) // self.sample_length

        for i in range(num_subsamples):
            subsample = signal[i * self.sample_length : (i + 1) * self.sample_length]
            if self.sample_length:
                subsample = subsample[: self.sample_length]  # Subsample the signal
            frequencias_picos = []
            X_signal = np.fft.fft(subsample)
            freqs_signal = np.fft.fftfreq(len(subsample), 1 / fs)  # Frequency bins
            x = freqs_signal[: len(freqs_signal) // 2][1:pontos]
            y = np.abs(X_signal)[: len(X_signal) // 2][1:pontos]
            threshold_height = 0.3 * max(y)
            peaks, _ = find_peaks(y, height=threshold_height)
            for peak in peaks:
                frequencias_picos.append(x[peak])
            results.append(frequencias_picos)
        concatenated_list = [item for sublist in results for item in sublist]

        return np.array(concatenated_list)

    def densidade_espectral_potencia(self, signal):
        results = []
        num_subsamples = len(signal) // self.sample_length

        for i in range(num_subsamples):
            subsample = signal[i * self.sample_length : (i + 1) * self.sample_length]
            if self.sample_length:
                subsample = subsample[: self.sample_length]  # Subsample the signal
            fft_result = np.fft.fft(subsample)
            psd = np.abs(fft_result) ** 2 / len(subsample)
            psd_init = psd[0:self.pontos]
            results.append(psd_init)

        return np.array(results)

    def fundamental_freq(self, signal, pontos=2000):
        results = []
        time = [
            1.953e-05,
            3.906e-05,
            5.859e-05,
            7.812e-05,
            9.766e-05,
            0.00011719,
            0.00013672,
        ]
        freq = np.fft.fftfreq(len(signal), time[1] - time[0])

        num_subsamples = len(signal) // self.sample_length
        for i in range(num_subsamples):
            subsample = signal[i * self.sample_length : (i + 1) * self.sample_length]
            if self.sample_length:
                subsample = subsample[: self.sample_length]  # Subsample the signal
            fundamental_freq = np.abs(
                freq[np.argmax(self.densidade_espectral_potencia(subsample, pontos))]
            )
            results.append(fundamental_freq)

        return np.array(results)

    def welchs_methods(self, signal):
        results = []
        num_subsamples = len(signal) // self.sample_length

        for i in range(num_subsamples):
            subsample = signal[i * self.sample_length : (i + 1) * self.sample_length]
            if self.sample_length:
                subsample = subsample[: self.sample_length]  # Subsample the signal
            frequencies, psd = welch(subsample, fs=50000, nperseg=256)
            results.append([frequencies, psd])

        return results

    def order_spectrum(self, signal, f_rot=13, fs=50000, percentage_magnitude=0.3):

        # Perform Fourier Transform
        n = len(signal)  # Length of the signal
        frequencies = np.fft.fftfreq(n, d=1 / fs)  # Frequency bins
        spectrum = np.abs(np.fft.fft(signal))  # Magnitude spectrum

        # Find peaks in the spectrum
        peaks, _ = find_peaks(
            spectrum[: n // 2], height=0
        )  # Find peaks in positive frequencies

        # Find magnitude of 1X component
        index_1x = np.argmax(spectrum[: n // 2])  # Index of maximum magnitude
        mag_1x = spectrum[index_1x]  # Magnitude of 1X component

        # List to store peak frequencies and magnitudes
        peak_list = []

        # Annotate orders and save peaks
        for peak in peaks:
            freq = frequencies[peak]
            order = int(round(freq / f_rot))
            if (
                spectrum[peak] > percentage_magnitude * mag_1x
            ):  # Highlight only if magnitude > 30% of 1X magnitude
                peak_list.append((freq, spectrum[peak]))

        return self.convert(peak_list)

    def convert(self, peak_list):
        # Convert list of tuples to numpy array
        array = np.array([list(t) for t in peak_list])

        # Transpose the array to make it one-dimensional with axis=1
        array = array.T
        return array

    def fft_envelope(self, signal):
        # Compute the FFT of the signal
        fft_signal = np.fft.fft(signal)

        # Compute the magnitude spectrum
        magnitude_spectrum = np.abs(fft_signal)

        # Apply the Hilbert transform to the magnitude spectrum
        analytic_signal = hilbert(magnitude_spectrum)

        # Extract the envelope from the analytic signal
        freq_envelope = np.abs(analytic_signal)

        return np.array(freq_envelope)

    def spectogram(self, signal, fs=50000):
        # Set parameters for the spectrogram
        window_length = int(0.025 * fs)  # 25 milliseconds window length
        overlap = int(0.01 * fs)  # 10 milliseconds overlap
        nfft = 512  # Number of FFT points
        frequencies, times, Sxx = spectrogram(
            signal, fs, nperseg=window_length, noverlap=overlap, nfft=nfft
        )
        # Plot the spectrogram
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), cmap="gray")
        # plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')  # Convert to dB for visualization
        # plt.xlabel('Time (s)')
        # plt.ylabel('Frequency (Hz)')
        # plt.title('Spectrogram of Sinusoidal Signal')

        # Turn off the axes
        plt.axis("off")

        # Save the spectrogram as an image file
        plt.savefig("spectrogram.png", bbox_inches="tight", pad_inches=0)