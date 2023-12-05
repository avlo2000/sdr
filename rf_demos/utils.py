import numpy as np
from matplotlib import pyplot as plt


def create_signal(amp: float, freq: float, phase: float, t: np.ndarray):
    return amp * np.sin(2 * np.pi * freq * t + phase)


def create_phase_modulated(amp: float, freq: float, phase: float, t: np.ndarray):
    n = len(t)
    p0 = n//4
    p1 = 2 * n//4
    p2 = 3 * n//4
    t_inter0 = t[:p0]
    t_inter1 = t[p0:p1]
    t_inter2 = t[p1:p2]
    t_inter3 = t[p2:]
    s0 = create_signal(amp, freq, phase, t_inter0)
    phase = np.arcsin(s0[-1] / amp) / (2 * np.pi * freq)
    s1 = create_signal(amp, freq, phase + 0.5 * np.pi, t_inter1)
    phase = np.arcsin(s1[-1] / amp) / (2 * np.pi * freq)
    s2 = create_signal(amp, freq, phase + 1.5 * np.pi, t_inter2)
    phase = np.arcsin(s2[-1] / amp) / (2 * np.pi * freq)
    s3 = create_signal(amp, freq, phase + np.pi, t_inter3)
    return np.concatenate([s0, s1, s2, s3])


def get_spec(sig: np.ndarray, sample_rate: float):
    n = len(sig)
    freqs = np.fft.fftfreq(n, d=1 / sample_rate)
    f2p = np.fft.fft(sig)
    return freqs, f2p


def plot_spec(sig: np.ndarray, sample_rate: float):
    n = len(sig)
    freqs = np.fft.fftfreq(n, d=1 / sample_rate)
    idx = np.argsort(freqs)[n//2:]
    f2p = np.fft.fft(sig)

    plt.plot(freqs[idx], np.abs(f2p[idx]))


def create_random_iq_ref(num_symbols):
    x_int = np.random.randint(0, 4, num_symbols)
    x_degrees = x_int * 360 / 4.0 + 45
    x_radians = x_degrees * np.pi / 180.0
    x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians)
    return x_symbols
