import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from utils import create_signal, plot_spec, create_phase_modulated, get_spec


def main():
    n = 1000
    time = np.linspace(0, 1.0, n)
    fs = n / time[1]

    modulated_freq = 5000
    carrier_freq = 50000
    info_sig = create_phase_modulated(1.0, modulated_freq, 0, time)
    carrier_sig = create_signal(1.0, carrier_freq, np.pi / 2, time)
    sig12 = info_sig * carrier_sig
    plt.subplot(811)
    plot_spec(info_sig, fs)
    plot_spec(carrier_sig, fs)

    plt.subplot(812)
    plt.plot(time, info_sig)
    plt.plot(time, carrier_sig)

    sos = signal.butter(10, carrier_freq, 'hp', fs=fs, output='sos')
    sig_tx = sig12  # signal.sosfilt(sos, sig12)

    plt.subplot(813)
    plot_spec(sig12, fs)
    plt.subplot(814)
    plot_spec(sig_tx, fs)
    plt.subplot(815)
    plt.plot(time, sig_tx)

    plt.subplot(816)
    air_sig = sig_tx * carrier_sig
    plot_spec(air_sig, fs)

    freqs, f2p_rx = get_spec(air_sig, fs)
    f2p_rx[abs(freqs) > carrier_freq + 5 * modulated_freq] = 0.0

    sos = signal.butter(10, carrier_freq + 5 * modulated_freq, 'lp', fs=fs, output='sos')
    filtered_air = signal.sosfilt(sos, air_sig)

    # filtered_air = np.fft.ifft(f2p_rx)
    plt.subplot(817)
    plot_spec(filtered_air, fs)

    plt.subplot(818)
    plt.plot(time, filtered_air)
    plt.show()


if __name__ == '__main__':
    main()
