from skimage.feature import hog, local_binary_pattern
from skimage import io, color
import numpy as np
import cv2


def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Extract HOG features
    features, _ = hog(gray_image, visualize=True)
    return features


def extract_lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Extract LBP features
    lbp_features = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    return lbp_features.flatten()


def extract_histogram(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_array = np.array(gray_image)
    histogram, _ = np.histogram(gray_array.flatten(), bins=256, range=(0, 256))
    return histogram


def extract_hist_mean_skewness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_array = np.array(gray_image)
    histogram, _ = np.histogram(gray_array.flatten(), bins=256, range=(0, 256))
    mean = np.mean(gray_array)
    normalized_histogram = histogram / float(np.sum(histogram))
    pixel_values = np.arange(256)
    skewness = np.sum(
        ((pixel_values - mean) / np.std(gray_array)) ** 3 * normalized_histogram
    )
    return [mean, skewness]
