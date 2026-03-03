import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft, spectrogram
import os
import numpy as np
import matplotlib.pyplot as plt
from .descriptors import (
    extract_hog_features,
    extract_lbp_features,
    extract_histogram,
    extract_hist_mean_skewness,
)
import imageio as io


class TimeFrequencyDomain:

    def __init__(self, sample_length=None, subsample = 25000):
        self.subsample = subsample
        self.sample_length = sample_length
        self.i_wt = 0
        self.i_wt = 0
        self.i_stft = 0
        self.channel = [3, 4, 5, 6, 7, 8]
        self.index_channel = 0
        self.label = 0
        self.duration = 1
        self.fs = 50000
        self.y = []
        self.classes = [
            "Normal",
            "Unb30g",
            "HorMis2mm",
            "VerMis127mm",
        ]

    def reset_label(self):
        self.label = 0

    def reset_inputs(self):
        self.X = []
        self.y = []

    def wavelet_transform_hog(self, signal, channel):
        self.index_channel = channel
        self.X = []
        c = self.i_wt % 4

        sample_length = self.fs * self.duration  # Total number of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples
            coefficients, frequencies = pywt.cwt(
                subsample, scales=np.arange(1, 128), wavelet="cmor"
            )

            plt.figure(figsize=(14, 10))
            plt.imshow(
                np.abs(coefficients),
                aspect="auto",
                cmap="jet",
                extent=[0, self.duration, 1, 128],
            )

            save_dir = "images_wt"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save the plot as an image
            image_path = os.path.join(
                save_dir,
                f"wt_image_channel_{self.channel[self.index_channel]}_{self.classes[c]}.png",
            )
            plt.axis("off")  # Turn off axis
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            hog_features = extract_hog_features(image)
            self.X.append(hog_features)
            os.remove(image_path)
        self.i_wt += 1

        return np.array(self.X)

    def wavelet_transform_lbp(self, signal, channel):
        self.X = []
        c = self.i_wt % 4
        self.index_channel = channel


        sample_length = self.fs * self.duration  # Total number of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples
            coefficients, frequencies = pywt.cwt(
                subsample, scales=np.arange(1, 128), wavelet="cmor"
            )

            plt.figure(figsize=(14, 10))
            plt.imshow(
                np.abs(coefficients),
                aspect="auto",
                cmap="jet",
                extent=[0, self.duration, 1, 128],
            )

            save_dir = "images_wt"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            image_path = os.path.join(
                save_dir,
                f"wt_image_channel_{self.channel[self.index_channel]}_{self.classes[c]}.png",
            )
            plt.axis("off")  # Turn off axis
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            lbp_features = extract_lbp_features(image)
            self.X.append(lbp_features)
            os.remove(image_path)
        self.i_wt += 1
        return np.array(self.X)

    def wavelet_transform_histogram(self, signal):
        self.X = []
        c = self.i_wt % 4

        sample_length = self.fs * self.duration  # Total number of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples
            coefficients, frequencies = pywt.cwt(
                subsample, scales=np.arange(1, 128), wavelet="cmor"
            )

            plt.figure(figsize=(14, 10))
            plt.imshow(
                np.abs(coefficients),
                aspect="auto",
                cmap="jet",
                extent=[0, self.duration, 1, 128],
            )

            save_dir = "images_wt"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save the plot as an image
            image_path = os.path.join(
                save_dir,
                f"wt_image_channel_{self.channel[self.index_channel]}_{self.classes[c]}.png",
            )
            plt.axis("off")  # Turn off axis
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            histogram = extract_histogram(image)
            self.X.append(histogram)

            os.remove(image_path)
        self.i_wt += 1
        return np.array(self.X)

    def wavelet_transform_hist_mean_skewness(self, signal):
        self.X = []
        c = self.i_wt % 4

        sample_length = self.fs * self.duration  # Total number of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples
            coefficients, frequencies = pywt.cwt(
                subsample, scales=np.arange(1, 128), wavelet="cmor"
            )

            plt.figure(figsize=(14, 10))
            plt.imshow(
                np.abs(coefficients),
                aspect="auto",
                cmap="jet",
                extent=[0, self.duration, 1, 128],
            )

            save_dir = "images_wt"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save the plot as an image
            image_path = os.path.join(
                save_dir,
                f"wt_image_channel_{self.channel[self.index_channel]}_{self.classes[c]}.png",
            )
            plt.axis("off")  # Turn off axis
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            for i in range(2):
                histogram = extract_hist_mean_skewness(image)
                self.X.append(histogram)

            os.remove(image_path)
        self.i_wt += 1

        return np.array(self.X)

    def spectrogram_hog(self, signal, channel):
        self.X = []
        self.index_channel = channel

        sample_length = self.fs * self.duration  # Total number of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples

            f, t, Sxx = spectrogram(subsample, self.fs)

            # return Sxx

            save_dir = f"images_spectogram/spectogram_images_channel_{self.channel[self.index_channel]}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plt.figure(figsize=(10, 6))
            plt.pcolormesh(t, f, Sxx, shading="gouraud")
            image_path = os.path.join(
                save_dir,
                f"spectogram_image_channel_{self.channel[self.index_channel]}.png",
            )
            plt.axis("off")  # Turn off axis
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            hog_features = extract_hog_features(image)
            self.X.append(hog_features)

            os.remove(image_path)
        self.i_wt += 1
        return np.array(self.X)

    def spectrogram_lbp(self, signal, channel):
        self.X = []
        self.index_channel = channel

        sample_length = self.fs * self.duration  # Total number of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples

            f, t, Sxx = spectrogram(subsample, self.fs)

            # return Sxx

            save_dir = f"images_spectogram/spectogram_images_channel_{self.channel[self.index_channel]}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plt.figure(figsize=(16, 10))
            plt.pcolormesh(t, f, Sxx, shading="gouraud")
            image_path = os.path.join(
                save_dir,
                f"spectogram_image_channel_{self.channel[self.index_channel]}.png",
            )
            plt.axis("off")  # Turn off axis
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            lbp_features = extract_lbp_features(image)
            self.X.append(lbp_features)

            os.remove(image_path)
        self.i_wt += 1
        return np.array(self.X)

    def spectrogram_histogram(self, signal, channel):
        self.X = []
        c = self.i_wt % 4
        self.index_channel = channel

        sample_length = self.fs * self.duration  # Total number of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples

            f, t, Sxx = spectrogram(subsample, self.fs)

            # return Sxx

            save_dir = f"images_spectogram/spectogram_images_channel_{self.channel[self.index_channel]}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plt.figure(figsize=(16, 10))
            plt.pcolormesh(t, f, Sxx, shading="gouraud")
            image_path = os.path.join(
                save_dir,
                f"spectogram_image_channel_{self.channel[self.index_channel]}_{self.classes[c]}.png",
            )
            plt.axis("off")  # Turn off axis
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            for i in range(2):
                lbp_features = extract_histogram(image)
                self.X.append(lbp_features)

            os.remove(image_path)
        self.i_wt += 1
        return np.array(self.X)

    def spectrogram_hist_mean_skewness(self, signal, channel):
        self.X = []
        c = self.i_wt % 4
        self.index_channel = channel

        sample_length = self.fs * self.duration  # Total number of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples

            f, t, Sxx = spectrogram(subsample, self.fs)

            # return Sxx

            save_dir = f"images_spectogram/spectogram_images_channel_{self.channel[self.index_channel]}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plt.figure(figsize=(16, 10))
            plt.pcolormesh(t, f, Sxx, shading="gouraud")
            image_path = os.path.join(
                save_dir,
                f"spectogram_image_channel_{self.channel[self.index_channel]}_{self.classes[c]}.png",
            )
            plt.axis("off")  # Turn off axis
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            for i in range(2):
                lbp_features = extract_hist_mean_skewness(image)
                self.X.append(lbp_features)

            os.remove(image_path)
        self.i_wt += 1
        return np.array(self.X)

    def short_time_fourier_transform_hog(self, signal, channel):
        self.X = []
        c = self.i_stft % 4
        self.index_channel = channel

        nperseg = 2048
        overlap = nperseg // 2
        sample_length = (
            self.fs * self.duration
        )  # Total nuself.mber of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples

            # Compute STFT
            frequencies, time, stft_response = stft(
                subsample, fs=self.fs, nperseg=nperseg, noverlap=overlap
            )

            # return stft_response
            # Save STFT response as an image
            save_dir = (
                f"images_stft/stft_images_channel_{self.channel[self.index_channel]}"
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            image_path = os.path.join(
                save_dir,
                f"stft_image_channel_{self.channel[self.index_channel]}_{self.classes[c]}.png",
            )
            plt.figure(figsize=(16, 10))
            plt.pcolormesh(time, frequencies, np.abs(stft_response), shading="gouraud")
            plt.axis("off")  # Remove axes
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            hog_features = extract_hog_features(image)
            self.X.append(hog_features)

            os.remove(image_path)
        self.i_wt += 1

        return np.array(self.X)

    def short_time_fourier_transform_lbp(self, signal, channel):
        self.X = []
        self.index_channel = channel
        c = self.i_stft % 4

        sample_rate = 50000  # 50 kHz
        nperseg = 2048
        overlap = nperseg // 2
        sample_length = (
            self.fs * self.duration
        )  # Total nuself.mber of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples

            # Compute STFT
            frequencies, time, stft_response = stft(
                subsample, fs=self.fs, nperseg=nperseg, noverlap=overlap
            )

            # return stft_response
            # Save STFT response as an image
            save_dir = (
                f"images_stft/stft_images_channel_{self.channel[self.index_channel]}"
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            image_path = os.path.join(
                save_dir,
                f"stft_image_channel_{self.channel[self.index_channel]}_{self.classes[c]}.png",
            )
            plt.figure(figsize=(16, 10))
            plt.pcolormesh(time, frequencies, np.abs(stft_response), shading="gouraud")
            plt.axis("off")  # Remove axes
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            lbp_features = extract_lbp_features(image)
            self.X.append(lbp_features)

            os.remove(image_path)
        self.i_stft += 1
        return np.array(self.X)

    def short_time_fourier_transform_histogram(self, signal, channel):
        self.X = []
        c = self.i_stft % 4
        self.index_channel = channel

        sample_rate = 50000  # 50 kHz
        nperseg = 2048
        overlap = nperseg // 2
        sample_length = (
            self.fs * self.duration
        )  # Total nuself.mber of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples

            # Compute STFT
            frequencies, time, stft_response = stft(
                subsample, fs=self.fs, nperseg=nperseg, noverlap=overlap
            )

            # return stft_response
            # Save STFT response as an image
            save_dir = (
                f"images_stft/stft_images_channel_{self.channel[self.index_channel]}"
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            image_path = os.path.join(
                save_dir,
                f"stft_image_channel_{self.channel[self.index_channel]}_{self.classes[c]}.png",
            )
            plt.figure(figsize=(16, 10))
            plt.pcolormesh(time, frequencies, np.abs(stft_response), shading="gouraud")
            plt.axis("off")  # Remove axes
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            lbp_features = extract_histogram(image)
            self.X.append(lbp_features)

            os.remove(image_path)
        self.i_stft += 1

        return np.array(self.X)

    def short_time_fourier_transform_hist_mean_skewness(self, signal, channel):
        self.X = []
        c = self.i_stft % 4
        self.index_channel = channel

        sample_rate = 50000  # 50 kHz
        nperseg = 2048
        overlap = nperseg // 2
        sample_length = (
            self.fs * self.duration
        )  # Total nuself.mber of samples in 5 seconds
        # sample_length=125000

        # Iterate over the signal in subsamples of 50,000 samples each
        for i in range(0, len(signal), sample_length):
            subsample = signal[
                i : i + sample_length
            ]  # Extract a subsample of 50,000 samples

            # Compute STFT
            frequencies, time, stft_response = stft(
                subsample, fs=self.fs, nperseg=nperseg, noverlap=overlap
            )

            # return stft_response
            # Save STFT response as an image
            save_dir = (
                f"images_stft/stft_images_channel_{self.channel[self.index_channel]}"
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            image_path = os.path.join(
                save_dir,
                f"stft_image_channel_{self.channel[self.index_channel]}_{self.classes[c]}.png",
            )
            plt.figure(figsize=(16, 10))
            plt.pcolormesh(time, frequencies, np.abs(stft_response), shading="gouraud")
            plt.axis("off")  # Remove axes
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            image = io.imread(f"{image_path}")
            lbp_features = extract_hist_mean_skewness(image)
            self.X.append(lbp_features)

            os.remove(image_path)
        self.i_stft += 1

        return np.array(self.X)
