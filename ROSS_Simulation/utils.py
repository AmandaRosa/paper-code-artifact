import math
import statistics
import numpy as np
import ross as rs
import scipy
from scipy.stats import skew, kurtosis
from scipy.stats import entropy
from scipy import signal as sig
from scipy.signal import butter, lfilter


def change_model_4dof_to_6dof(rotor_model, alpha=0.0, beta=0.0):
    """
    Defines a function to change models from 4 degrees of freedom to 6

    Parameters
    ----------
        rotor_model: obj / rs.Rotor
            Instance of an object of the Rotor class, preferably implemented through the ROSS library
    """
    # Updates the shaft elements
    shaft_elements_lenght = rotor_model.shaft_elements
    shaft_elem = [
        rs.ShaftElement6DoF(
            material=rotor_model.shaft_elements[l].material,
            L=rotor_model.shaft_elements[l].L,
            n=rotor_model.shaft_elements[l].n,
            idl=rotor_model.shaft_elements[l].idl,
            odl=rotor_model.shaft_elements[l].odl,
            idr=rotor_model.shaft_elements[l].idr,
            odr=rotor_model.shaft_elements[l].odr,
            alpha=alpha,
            beta=beta,
            rotary_inertia=True,
            shear_effects=True,
        )
        for l, p in enumerate(shaft_elements_lenght)
    ]
    # Updates the discs
    disk_elem = [
        rs.DiskElement6DoF(
            n=rotor_model.disk_elements[l].n,
            m=rotor_model.disk_elements[l].m,
            Id=rotor_model.disk_elements[l].Id,
            Ip=rotor_model.disk_elements[l].Ip,
        )
        for l in range(len(rotor_model.disk_elements))
    ]
    # Updates the bearings
    bearing_elem = [
        rs.BearingElement6DoF(
            n=rotor_model.bearing_elements[l].n,
            kxx=rotor_model.bearing_elements[l].kxx,
            kyy=rotor_model.bearing_elements[l].kyy,
            cxx=rotor_model.bearing_elements[l].cxx,
            cyy=rotor_model.bearing_elements[l].cyy,
            frequency=rotor_model.bearing_elements[l].frequency,
        )
        for l in range(len(rotor_model.bearing_elements))
    ]
    # Returns the output of function
    return rs.Rotor(shaft_elem, disk_elem, bearing_elem)


def get_skewness(signal):
    return skew(signal)


def get_kurtosis(signal):
    return kurtosis(signal)


def get_shape_factor(signal):
    N = len(signal)
    return np.sqrt(((signal**2).sum() / N) / ((abs(signal)).sum() / N))


def get_variance(signal):
    return statistics.variance(signal)


def get_std(signal):
    return statistics.stdev(signal)


def get_rms_acceleration(signal):
    N = len(signal)
    return np.sqrt(1 / N * (signal**2).sum())


def get_peak_acceleration(signal):
    return max(abs(signal))


def get_crest_factor(signal):
    return get_peak_acceleration(signal) / get_rms_acceleration(signal)


def get_frequency_centre(signal):
    return ((np.diff(signal)).sum()) / (2 * np.pi * np.sum(signal * 2))


def get_frequency(yf, xf):
    return xf[np.argmax(yf)]


def get_x_freq_val(yf, xf):
    f_x = np.abs(xf[np.argmax(yf)])
    return (
        (yf[np.where(xf == (f_x))[0][0]]) / max(yf),
        (f_x),
    )


def get_x2_freq_val(yf, xf):
    # deltaf = xf[1] - xf[0]
    f_x1 = np.argmax(yf) + 1
    offset = len(yf[:f_x1])
    f_x = np.abs(xf[np.argmax(yf[f_x1:]) + offset])
    return (
        (yf[np.where(xf == (f_x))[0][0]]) / max(yf),
        (f_x),
    )


def get_x3_freq_val(yf, xf):
    # deltaf = xf[1] - xf[0]
    ## função para pegar a posição do terceiro maior pico do sinal em yf
    try:
        f_x1 = np.argmax(yf) + 1
        offset = len(yf[:f_x1])
        f_x2 = np.argmax(yf[f_x1:]) + offset + 1
        offset2 = len(yf[:f_x2])
        f_x = np.abs(xf[np.argmax(yf[f_x2:]) + offset2])

        return (
            (yf[np.where(xf == (f_x))[0][0]]) / max(yf),
            (f_x),
        )
    except:
        return (0, 0)


def reconstruir_sinal(fft_real):
    """
    Reconstrói um sinal no tempo a partir da parte real da FFT.

    Argumentos:
      fft_real: A parte real da FFT do sinal.

    Retorna:
      sinal_reconstruido: O sinal reconstruído no tempo.
    """
    # fft_invertido = fft_real.copy()
    # fft_invertido = np.flip(fft_invertido)
    # fft_real = np.concatenate((fft_real, fft_invertido))

    # Cria a parte imaginária da FFT a partir da parte real.
    fft_imag = np.zeros_like(fft_real)

    # A primeira e última frequência da parte imaginária são zero.
    fft_imag[0] = 0
    fft_imag[-1] = 0

    # As frequências negativas são o conjugado complexo das frequências positivas.
    for i in range(1, len(fft_real) // 2):
        fft_imag[i] = -fft_imag[-i]

    # Combina as partes real e imaginária para formar a FFT completa.
    fft_completa = fft_real + 1j * fft_imag

    # Realiza a inversa da FFT para obter o sinal no tempo.
    sinal_reconstruido = np.fft.ifft(fft_completa)

    return sinal_reconstruido


# function to get all positions of local peaks and their values that are above a certain threshold
# in the frequency domain
def get_all_peaks(yf, xf, threshold, refratory=10):
    peaks = []
    i = 0
    while i < len(yf) - refratory:
        if (
            yf[i] > yf[i - refratory]
            and yf[i] > yf[i + refratory]
            and yf[i] > threshold
        ):
            peaks.append((yf[i], xf[i]))
            i += refratory
        i += 1
    # get only the 5 highest peaks
    # if not 5 peaks, fill with zeros
    ret_aux = np.zeros((5, 2))
    for i in range(5):
        if i < len(peaks):
            ret_aux[i] = peaks[i]

    return ret_aux


def get_all_peaks_(yf, xf, threshold, refratory=10):
    peaks = []
    i = 0
    while i < len(yf) - refratory:
        # get the position of the maximum value in the window of refratory
        max_pos = np.argmax(yf[i : i + refratory])
        if yf[i + max_pos] > threshold and (
            (peaks and abs(peaks[-1][1] - xf[i + max_pos]) > refratory) or not peaks
        ):
            peaks.append((yf[i + max_pos], xf[i + max_pos]))
        i += refratory + 1
        # i += 1
    # get only the 5 highest peaks
    # if not 5 peaks, fill with zeros
    ret_aux = np.zeros((5, 2))
    for i in range(5):
        if i < len(peaks):
            ret_aux[i] = peaks[i]

    return ret_aux


# make fft of signal
def get_fft(signal):
    N = len(signal)
    yf = scipy.fftpack.fft(signal)
    return np.abs(yf)


def get_zero_crossing(signal):
    return len(np.where(np.diff(np.sign(signal)))[0])


def get_frequencia_sinal(signal):
    # get the frequency of the signal using zero crossing
    return get_zero_crossing(signal) / (2 * len(signal))


def get_media_ponderada_sinal(signal, fa, rpm, qtde_divisoes):
    tam_periodo = qtde_divisoes * int(fa / rpm)
    qtde_partes = int(len(signal) / tam_periodo)
    sinal_base = []
    for i in range(qtde_partes):
        sinal_base.append(signal[i * tam_periodo : (i + 1) * tam_periodo])

    return sinal_base


def fft_welch(x, freq_aquis, nraias, noverlap=0, janela="hanning", freq_range=[0, 100]):
    if noverlap <= 0 or noverlap >= 1:  # noverlap é definido entre 0 e 1
        noverlap = None
    if janela == "hanning":
        janela = "hann"
    if janela == "retangular":
        janela = "boxcar"

    # definir nfft e overlap
    nfft = round(2.56 * nraias)
    noverlap = round(noverlap * nfft) if noverlap is not None else noverlap

    # construir janela e achar fator de escala do espectro
    window = sig.get_window(janela, nfft)
    freq, dep = sig.welch(
        x,
        freq_aquis,
        window=window,
        noverlap=noverlap,
        nperseg=nfft,
        nfft=None,
        detrend="constant",
        return_onesided=True,
        scaling="spectrum",
        axis=-1,
        average="mean",
    )

    # cortar o espectro para o número de raias desejado
    if freq_range is not None:
        dep = dep[(freq >= freq_range[0]) & (freq <= freq_range[1])]
        freq = freq[(freq >= freq_range[0]) & (freq <= freq_range[1])]

    return np.sqrt(dep), freq


def get_fft_visual(x, range_freq=[0, 100], dt=2e-5):

    b = np.floor(len(x) / 2)
    c = len(x)
    df = 1 / (c * dt)
    try:
        x_amp = scipy.fft.fft(x.values)[: int(b)]
    except:
        x_amp = scipy.fft.fft(x)[: int(b)]
    x_amp = x_amp * 2 / c
    x_phase = np.angle(x_amp)
    x_amp = np.abs(x_amp)

    freq = np.arange(0, df * b, df)
    freq = freq[: int(b)]  # Frequency vector

    if range_freq is not None:
        x_amp = x_amp[(freq >= range_freq[0]) & (freq <= range_freq[1])]
        freq = freq[(freq >= range_freq[0]) & (freq <= range_freq[1])]

    # normalizar o x_amp de zero a 1
    # x_amp = (x_amp - min(x_amp)) / (max(x_amp) - min(x_amp))

    # # muitos sinais estão começando acima de zero, então vamos diminuir o primeiro valor do sinal todo e o que for menor que zero será zero
    # x_amp = x_amp - x_amp[0]
    # x_amp[x_amp < 0] = 0

    return x_amp, freq


def get_fft_visual_data_augumentation(x, range_freq=[0, 100], dt=2e-5):

    b = np.floor(len(x) / 2)
    b_completo = np.floor(len(x))
    c = len(x)
    df = 1 / (c * dt)
    try:
        x_amp = scipy.fft.fft(x.values)[: int(b)]
        x_amp_completo = scipy.fft.fft(x.values)[: int(b_completo)]
    except:
        x_amp = scipy.fft.fft(x)[: int(b)]
        x_amp_completo = scipy.fft.fft(x)[: int(b_completo)]

    x_amp = x_amp * 2 / c
    x_phase = np.angle(x_amp)
    x_amp = np.abs(x_amp)

    freq = np.arange(0, df * b, df)
    freq = freq[: int(b)]  # Frequency vector
    index_freq_completo = np.arange(0, df * b_completo, df)

    # freq_invertido = freq.copy()
    # freq_invertido = np.flip(freq_invertido)
    # freq_completo = np.concatenate((-freq_invertido, freq))

    if range_freq is not None:
        x_amp = x_amp[(freq >= range_freq[0]) & (freq <= range_freq[1])]
        freq = freq[(freq >= range_freq[0]) & (freq <= range_freq[1])]

    # normalizar o x_amp de zero a 1
    # x_amp = (x_amp - min(x_amp)) / (max(x_amp) - min(x_amp))

    # normalizar o x_amp_completo de zero a 1
    # x_amp_completo = (x_amp_completo - min(x_amp_completo)) / (
    #     max(x_amp_completo) - min(x_amp_completo)
    # )

    # # muitos sinais estão começando acima de zero, então vamos diminuir o primeiro valor do sinal todo e o que for menor que zero será zero
    # x_amp = x_amp - x_amp[0]
    # x_amp[x_amp < 0] = 0

    return x_amp, freq, x_amp_completo, index_freq_completo


def normaliza_zero_um(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10) + 1e-10


def processing_fft_signal(signal):
    # signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    media = np.mean(signal)
    variacao = np.std(signal)
    signal[signal < 0.3 * media] = 0.3 * media
    #
    signal = np.log10(signal)
    # signal[0] = signal[1]
    signal = normaliza_zero_um(signal)
    return signal


def average_signal(signal, num_parts):
    """
    Divide o sinal em partes e calcula a média ponto a ponto.

    Parâmetros:
        signal (numpy.ndarray): Sinal de entrada.
        num_parts (int): Número de partes em que o sinal será dividido.

    Retorna:
        numpy.ndarray: Média ponto a ponto do sinal dividido.
    """
    if len(signal) % num_parts != 0:
        signal = signal[: len(signal) - (len(signal) % num_parts)]
    # Verifica se o número de partes é válido
    if num_parts <= 0 or num_parts >= len(signal):
        raise ValueError(
            "O número de partes deve ser maior que 0 e menor que o comprimento do sinal."
        )

    # Divide o sinal em partes iguais
    parts = np.array_split(signal, num_parts)

    # Calcula a média ponto a ponto
    averaged_signal = np.mean(parts, axis=0)
    averaged_signal = averaged_signal - np.mean(averaged_signal)

    return averaged_signal


def filtro_passa_baixa(sinal, cutoff, fs, order=5):
    """
    Aplica um filtro passa-baixa no sinal.

    Parâmetros:
        sinal (numpy.ndarray): Sinal de entrada.
        cutoff (float): Frequência de corte do filtro.
        fs (float): Frequência de amostragem do sinal.
        order (int): Ordem do filtro.

    Retorna:
        numpy.ndarray: Sinal filtrado.
    """
    from scipy.signal import butter, lfilter

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = lfilter(b, a, sinal)
    return y


def filtro_passa_alta(sinal, cutoff, fs, order=5):
    """
    Aplica um filtro passa-alta no sinal.

    Parâmetros:
        sinal (numpy.ndarray): Sinal de entrada.
        cutoff (float): Frequência de corte do filtro.
        fs (float): Frequência de amostragem do sinal.
        order (int): Ordem do filtro.

    Retorna:
        numpy.ndarray: Sinal filtrado.
    """

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    y = lfilter(b, a, sinal)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def filtro_passa_banda(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def reduzir_sinal(sinal, novo_tamanho):
    tamanho_original = len(sinal)
    fator_reducao = tamanho_original // novo_tamanho
    novo_sinal = np.zeros(novo_tamanho)

    for i in range(novo_tamanho):
        inicio = i * fator_reducao
        fim = (i + 1) * fator_reducao
        novo_sinal[i] = np.mean(sinal[inicio:fim])

    return novo_sinal


def redimensionar_sinal(sinal, novo_tamanho):
    ## função para redimensionar o sinal para maior ou menor por interpolacao
    ## novo_tamanho é o tamanho do sinal após a interpolação
    ## sinal é o sinal a ser redimensionado
    ## retorna o sinal redimensionado
    from scipy.interpolate import interp1d

    x = np.arange(0, len(sinal))
    f = interp1d(x, sinal)
    xnew = np.linspace(0, len(sinal) - 1, novo_tamanho)
    return f(xnew)


def reconstruct_fft(real_part):
    n = len(real_part)  # Comprimento do sinal

    # Para a parte real, os valores são simétricos, exceto o primeiro e último elementos
    real_part_symmetric = np.concatenate((real_part, real_part[-2:0:-1]))

    # A parte imaginária será inicialmente preenchida com zeros
    imaginary_part = np.zeros_like(real_part_symmetric)

    # Para reconstruir a parte imaginária, fazemos a transformação de Hilbert na parte real
    # que é equivalente a multiplicar a FFT pela função de janela complexa
    # Janela complexa = [1, 2, 2, ..., 2, 1]
    imaginary_part[1:n] = 2 * np.cumsum(real_part_symmetric[1:n])

    return real_part_symmetric + 1j * imaginary_part


def CalculaRotac(x, freq_aquis, freq_ini=1, freq_fim=2000):
    # Esta função recebe o vetor com o sinal no domínio do tempo x e estima a
    # velocidade de rotação da máquina frec_rotac em Hz

    # inicialização

    m = 1
    freq = freq_aquis
    freq_nyquist = 0.5 * freq_aquis
    x_reamostrado = np.copy(x)
    npto = len(x)
    # Recondicionar o sinal via filtros passa - baixas
    while freq > 500:
        freq = freq / 2
        m = m * 2
    dt = m / freq_aquis
    N = int(npto / m)  # acertar dimensões
    npto1 = N * m
    if npto1 != npto:
        npto = npto1
        x = x[0:npto]
    if m > 1:  # Filtrar e re-amostrar
        Wn = np.array([100 / freq_nyquist])  # Filtro passa baixas em 100 Hz
        sos = sig.butter(N=2, Wn=Wn, btype="lowpass", analog=False, output="sos")
        x_filtrado = sig.sosfilt(sos, x)
        x_filtrado = sig.sosfilt(sos, x_filtrado)
        x_filtrado = sig.sosfilt(sos, x_filtrado)
        x_reamostrado = np.zeros((m, N))
        for i in range(m):
            x_reamostrado[i, :] = x_filtrado[i:npto:m]

    r = np.zeros((m, int(2 * N) - 1))
    for i in range(m):
        r[i, :] = np.correlate(x_reamostrado[i, :], x_reamostrado[i, :], mode="full")
    r = r[:, N:0:-1]
    exp4 = 1.0 / np.exp(np.linspace(0, (N - 1) * dt, num=N) * 4)
    r = r * exp4
    z = np.zeros(npto)
    z[0:N] = np.mean(r, axis=0)
    df = 1 / (npto * dt)
    Z = np.abs(m * np.fft.fft(z))
    i_ini = int(freq_ini / df)
    i_fim = int(freq_fim / df)
    posic = np.argmax(Z[i_ini:i_fim])
    freq_rotac = freq_ini + posic * df
    return freq_rotac


# Estimate power spectral density using Welch’s method
def get_psd(spectral):
    return np.sqrt(np.sum(np.abs(spectral) ** 2) / len(spectral))


# Estimate the density function of a signal
def get_density(spectral):
    return np.sum(np.abs(spectral)) / len(spectral)


def get_entropy(signal):
    return entropy(signal)


def get_energy(signal):
    return np.sum(np.abs(signal) ** 2)


def get_energy_ratio(signal):
    return np.sum(np.abs(signal) ** 2) / np.sum(np.abs(signal))


def SKEWNESS(x):
    # dado o vetor x, esta função retorna uma estimativa da assimetria de x
    xb = np.mean(x)
    k2 = np.mean(np.power(x - xb, 2))
    k3 = np.mean(np.power(x - xb, 3))
    x_k = 0
    if k2 > 1e-16:
        x_k = k3 / k2 ** (3 / 2)
    return x_k


def FREQUENCIA_CENTRAL(x):
    # dado o vetor x, esta função retorna uma estimativa da frequência central de x
    xb = np.mean(x)
    k2 = np.mean(np.power(x - xb, 2))
    k3 = np.mean(np.power(x - xb, 3))
    k4 = np.mean(np.power(x - xb, 4))
    k5 = np.mean(np.power(x - xb, 5))
    k6 = np.mean(np.power(x - xb, 6))
    fc = 0
    if k2 > 1e-16:
        fc = (k3 * k5 - k4 * k4) / (k2 * k2 * k2)
    return fc


def KURTOSIS(x):
    # dado o vetor x, esta função retorna uma estimativa da Curtose de x
    xb = np.mean(x)
    k2 = np.mean(np.power(x - xb, 2))
    k4 = np.mean(np.power(x - xb, 4))
    k = 0
    if k2 > 1e-16:
        k = k4 / k2 ** (2)
    return k


def RMS(x):
    # dado o vetor x, esta função retorna o seu valor RMS
    dep = np.array(x)
    den = len(x)
    if den < 2:  # correção para compensar o efeito do ganho da janela
        nraias = len(dep)
        area = dep[nraias - 1] ** 2 / dep[nraias - 2] ** 2
        den = 1.0 / area
    rms = 0
    if den > 0:
        rms = np.sqrt(np.dot(x[1], x[1]) / den)
    return rms


def MAX(x):
    # dado o vetor x, esta função retorna o valor máximo dos valores absolutos de x
    x_max = np.max(np.abs(x))
    return x_max


def PICO(x):
    # dado o vetor x, esta função retorna o valor pico-a-pico de x
    x_pico = np.ptp(x)
    return x_pico


def K4(x):
    # dado o vetor x, esta função retorna uma estimativa do K4 de x
    k4 = RMS(x) * KURTOSIS(x)
    return k4


def K6(x):
    # dado o vetor x, esta função retorna uma estimativa do K6 de x
    xb = np.mean(x)
    k2 = np.mean(np.power(x - xb, 2))
    k6 = 0
    if k2 > 1e-16:
        k6 = np.mean(np.power(x - xb, 6))
        k6 /= k2**3
    k6 *= RMS(x)
    return k6


def FC(x):
    # dado o vetor x, esta função retorna uma estimativa do fator de crista de x
    fc = 0
    den = RMS(x)
    num = MAX(x)
    if den > 1e-16 and num > 1e-16:
        fc = 20 * math.log10(num / den)
    return fc


def gaussian_noise(data, mean, std):
    noise = np.random.normal(mean, std, len(data))
    return data + noise


def normalizar_sinal(sinal):
    sinal = sinal - np.mean(sinal)
    sinal = sinal / np.max(np.abs(sinal))
    return sinal


def normalizar_sinal_menos_um_a_um(sinal):
    # função para normalizar o sinal entre -1 e 1
    min_val = np.min(sinal)
    max_val = np.max(sinal)
    normalized_signal = 2 * (sinal - min_val) / (max_val - min_val) - 1
    return normalized_signal - np.mean(normalized_signal)


def average_signal(signal, num_parts):
    """
    Divide o sinal em partes e calcula a média ponto a ponto.

    Parâmetros:
        signal (numpy.ndarray): Sinal de entrada.
        num_parts (int): Número de partes em que o sinal será dividido.

    Retorna:
        numpy.ndarray: Média ponto a ponto do sinal dividido.
    """
    if len(signal) % num_parts != 0:
        signal = signal[: len(signal) - (len(signal) % num_parts)]
    # Verifica se o número de partes é válido
    if num_parts <= 0 or num_parts >= len(signal):
        raise ValueError(
            "O número de partes deve ser maior que 0 e menor que o comprimento do sinal."
        )

    # Divide o sinal em partes iguais
    parts = np.array_split(signal, num_parts)

    # Calcula a média ponto a ponto
    averaged_signal = np.mean(parts, axis=0)
    averaged_signal = averaged_signal - np.mean(averaged_signal)

    return averaged_signal


# Função para calcular a Distância Euclidiana
def euclidean_distance(fft1, fft2):
    return np.linalg.norm(fft1 - fft2)


# Função para calcular a Correlacao de Pearson
def pearson_correlation(fft1, fft2):
    return np.corrcoef(fft1, fft2)[0, 1]
