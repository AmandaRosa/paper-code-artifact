from scipy.stats import skew, kurtosis
from scipy.signal import hilbert
import numpy as np
import statistics


class TimeDomain:

    def __init__(self, subsample=2500, sample_length=None):
        self.subsample = subsample
        self.sample_length = sample_length

    def skewness(self, signal):
        return np.asarray(
            [
                skew(signal[i : i + self.subsample])
                for i in range(0, len(signal), self.subsample)
            ],
            dtype=np.float64,
        )

    def kurtosis(self, signal):
        return np.asarray(
            [
                kurtosis(signal[i : i + self.subsample])
                for i in range(0, len(signal), self.subsample)
            ],
            dtype=np.float64,
        )

    def shape_factor(self, signal):
        signal_array = np.asarray(signal)  # Convert signal to a NumPy array
        N = len(signal_array)
        return np.asarray(
            [
                np.sqrt(
                    ((signal_array[i : i + self.subsample] ** 2).sum() / self.subsample)
                    / (
                        (np.abs(signal_array[i : i + self.subsample])).sum()
                        / self.subsample
                    )
                )
                for i in range(0, N, self.subsample)
            ],
            dtype=np.float64,
        )

    def variance(self, signal):
        return np.asarray(
            [
                (
                    statistics.variance(signal[i : i + self.subsample])
                    if len(signal[i : i + self.subsample]) > 1
                    else 0.0
                )
                for i in range(0, len(signal), self.subsample)
            ],
            dtype=np.float64,
        )

    def std(self, signal):
        return np.asarray(
            [
                (
                    statistics.stdev(signal[i : i + self.subsample])
                    if len(signal[i : i + self.subsample]) > 1
                    else 0.0
                )
                for i in range(0, len(signal), self.subsample)
            ],
            dtype=np.float64,
        )

    def standard_error(self, signal):
        return np.asarray(
            [
                statistics.stdev(signal[i : i + self.subsample])
                / np.sqrt(len(signal[i : i + self.subsample]))
                for i in range(0, len(signal), self.subsample)
            ],
            dtype=np.float64,
        )

    def peak_acceleration(self, signal):
        if isinstance(signal, list):
            signal = np.asarray(signal)

        if signal.ndim > 1:
            signal = signal.flatten()

        return np.asarray(
            [
                np.max(np.abs(signal[i : i + self.subsample]))
                for i in range(0, len(signal), self.subsample)
            ],
            dtype=np.float64,
        )

    def rms_acceleration(self, signal):
        signal_array = np.asarray(signal)
        N = len(signal_array)
        rms_values = []
        for i in range(0, N - self.subsample + 1, self.subsample):
            subsample_signal = signal_array[i : i + self.subsample]
            rms_value = np.sqrt(np.mean(subsample_signal**2))
            rms_values.append(rms_value)
        return np.asarray(rms_values, dtype=np.float64)

    # Also measures the ratio of the peak amplitude of a signal to its RMS value. It indicates how "peaky" or transient the signal is, with higher crest factors indicating larger peak values relative to the average amplitude.
    def crest_factor(self, signal):
        rms_values = self.rms_acceleration(signal)
        peak_values = self.peak_acceleration(signal)

        crest_factors = []
        for i in range(len(rms_values)):
            if rms_values[i] != 0:
                crest_factor = peak_values[i] / rms_values[i]
            else:
                crest_factor = float("inf")
            crest_factors.append(crest_factor)

        return np.asarray(crest_factors, dtype=np.float64)

    def mean_value(self, signal):
        return np.asarray(
            [
                np.mean(signal[i : i + self.subsample])
                for i in range(0, len(signal), self.subsample)
            ],
            dtype=np.float64,
        )

    def root_mean_square(self, signal):
        return np.asarray(
            [
                np.sqrt(np.mean(np.square(signal[i : i + self.subsample])))
                for i in range(0, len(signal), self.subsample)
            ],
            dtype=np.float64,
        )

    def peak_to_peak_amplitude(self, signal):
        return np.asarray(
            [
                np.ptp(signal[i : i + self.subsample])
                for i in range(0, len(signal), self.subsample)
            ],
            dtype=np.float64,
        )

    def zero_crossing(self, signal):
        zero_crossings = []

        for i in range(0, len(signal) - self.subsample + 1, self.subsample):
            sub_signal = signal[i : i + self.subsample]
            crossings = np.where(np.diff(np.sign(sub_signal)))[0]
            zero_crossings.append(len(crossings))

        return np.array(zero_crossings)

    def wavelength(
        self, signal
    ):  ## method calculates the wavelength for each sub-sample of a signal based on the number of zero crossings
        zero_crossings = self.zero_crossing(signal)
        return np.asarray(
            [
                len(signal[i : i + self.subsample]) / (crossing + 1e-10)
                for i, crossing in zip(
                    range(0, len(signal), self.subsample), zero_crossings
                )
            ],
            dtype=np.float64,
        )

    def wilson_amplitude(self, signal):
        return np.asarray(
            [
                (
                    np.max(signal[i : i + self.subsample])
                    - np.min(signal[i : i + self.subsample])
                )
                / 2
                for i in range(0, len(signal), self.subsample)
            ],
            dtype=np.float64,
        )

    def impulse_factor(
        self, signal
    ):  # The impulse factor is a measure used in signal processing to characterize the shape of a signal and its tendency towards impulse-like behavior. It quantifies the ratio of the peak amplitude to the mean amplitude within a segment of the signal.
        return np.asarray(
            [
                np.max(np.abs(signal[i : i + self.subsample]))
                / np.abs(np.mean(signal[i : i + self.subsample]))
                for i in range(0, len(signal), self.subsample)
            ],
            dtype=np.float64,
        )

    ###  Measures the margin or headroom within a signal relative to its mean square value. It quantifies how much additional capacity or strength is available beyond what is required for the signal.
    def margin_factor(
        self, signal
    ):  # The margin factor is a measure that quantifies the margin or headroom within a signal relative to its mean square value.
        margin_factors = [
            np.max(np.abs(signal[i : i + self.subsample]))
            / np.sqrt(np.mean(np.square(signal[i : i + self.subsample])))
            for i in range(0, len(signal), self.subsample)
        ]
        return np.asarray(margin_factors, dtype=np.float64)

    # Measures the ratio of the peak amplitude of a signal to its root mean square (RMS) value. It provides insights into the severity of peaks or impulses in the signal relative to its overall level.
    def clearance_factor(self, signal):
        result = []
        for i in range(0, len(signal), self.subsample):
            sub_signal = signal[i : i + self.subsample]
            rms_value = np.sqrt(np.mean(np.square(sub_signal)))
            result.append(float(np.max(np.abs(sub_signal)) / rms_value))

        result = np.array(result)
        return result

    def histogram(self, signal, bins=10):
        histograms = [
            np.histogram(signal[i : i + self.subsample], bins=bins)
            for i in range(0, len(signal), self.subsample)
        ]
        counts = np.asarray([hist[0] for hist in histograms], dtype=np.float64)
        bin_edges = np.asarray([hist[1] for hist in histograms], dtype=np.float64)
        return counts, bin_edges

    def entropy(self, signal):
        histograms, _ = self.histogram(signal)
        probabilities = histograms / np.sum(histograms, axis=1, keepdims=True)
        return np.asarray(
            [-np.sum(prob * np.log2(prob + 1e-10)) for prob in probabilities],
            dtype=np.float64,
        )

    def envelope(self, signal):
        envelope = np.abs(hilbert(signal))
        return envelope

    def envelope_from_subsamples(self, signal):
        num_samples = len(signal) / self.subsample
        envelopes = []

        for i in range(int(num_samples)):
            if i == 0:
                start = 0
                end = start + self.subsample
            else:
                start = end
                end = start + self.subsample
            subsample_signal = signal[start:end]
            analytic_signal = hilbert(subsample_signal)
            envelope = np.abs(analytic_signal)
            envelopes.append(envelope)

        return np.array(envelopes)

    def moving_average_from_envelope(self, signal):
        num_samples = len(signal) / self.subsample
        rolling_mean_from_envelope = []

        for i in range(int(num_samples)):
            if i == 0:
                start = 0
                end = start + self.subsample
            else:
                start = end
                end = start + self.subsample
            subsample_signal = signal[start:end]
            analytic_signal = hilbert(subsample_signal)
            envelope = np.abs(analytic_signal)
            window_size = 1000
            rolling_mean = np.convolve(
                envelope, np.ones(window_size) / window_size, mode="same"
            )
            rolling_mean_from_envelope.append(rolling_mean)

        return np.array(rolling_mean_from_envelope)

    def moving_average_from_signal(self, signal):
        num_samples = len(signal) / self.subsample
        rolling_mean_from_signal = []

        for i in range(int(num_samples)):
            if i == 0:
                start = 0
                end = start + self.subsample
            else:
                start = end
                end = start + self.subsample
            subsample_signal = signal[start:end]
            window_size = 1000
            rolling_mean = np.convolve(
                subsample_signal, np.ones(window_size) / window_size, mode="same"
            )
            rolling_mean_from_signal.append(rolling_mean)

        return np.array(rolling_mean_from_signal)

    def average_mean_from_envelope(self, signal):
        num_full_subsamples = len(signal) // self.subsample

        means = []
        for i in range(num_full_subsamples):
            start_index = i * self.subsample
            end_index = (i + 1) * self.subsample
            subsample = signal[start_index:end_index]
            envelope_subsample = self.envelope(subsample)
            mean_subsample = np.mean(envelope_subsample)
            means.append(mean_subsample)

        means = np.array(means)

        return means