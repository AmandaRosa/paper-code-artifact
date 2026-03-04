import ross as rs
import numpy as np
from ross.units import Q_, check_units
import scipy
from utils import *
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

rotor_file = "lmest_bancada_2discos.toml"


# make fft of signal
def get_fft(signal):
    N = len(signal)
    yf = scipy.fftpack.fft(signal)
    return np.abs(yf)


# make fft of signal
def get_fft_2(signal):
    N = len(signal)
    yf = scipy.fft.fft(signal)
    return np.abs(yf)


# make fft of signal
def get_fft_3(x):
    b = np.floor(len(x) / 2)
    c = len(x)

    x_amp = scipy.fft.fft(x)[: int(b)]
    x_amp = x_amp * 2 / c
    x_amp = np.abs(x_amp)

    return x_amp


def get_fft_4(x, range_freq=[0, 100]):
    dt = 4e-4
    b = np.floor(len(x) / 2)
    c = len(x)
    df = 1 / (c * dt)

    x_amp = scipy.fft.fft(x)[: int(b)]
    x_amp = x_amp * 2 / c
    x_phase = np.angle(x_amp)
    x_amp = np.abs(x_amp)

    freq = np.arange(0, df * b, df)
    freq = freq[: int(b)]  # Frequency vector

    return x_amp, freq


def run(
    dt, tf, probes, speed, unbalance_phase, unbalance_magnitude, missalignment_angle
):
    rotor = rs.Rotor.load(rotor_file)

    
    magunbt = np.array([unbalance_magnitude, unbalance_magnitude])
    phaseunbt = np.array([unbalance_phase, unbalance_phase])

    # alpha and beta are the proportional damping coefficients
    rotor = change_model_4dof_to_6dof(rotor, alpha=1, beta=1e-4)

    misalignment = rotor.run_misalignment(
        coupling="flex",
        dt=dt,
        tI=0,
        tF=tf,
        kd=40 * 10 ** (3),
        ks=38 * 10 ** (3),
        eCOUPx=2 * 10 ** (-4),
        eCOUPy=2 * 10 ** (-4),
        misalignment_angle=missalignment_angle,
        TD=0,
        TL=0,
        n1=0,
        speed=Q_(speed, "RPM"),
        unbalance_magnitude=magunbt,
        unbalance_phase=phaseunbt,
        mis_type="parallel",
        print_progress=False,
    )

    fig, sinal_normal_recons = misalignment.plot_dfft(probe=probes, yaxis_type="log")
    return sinal_normal_recons


