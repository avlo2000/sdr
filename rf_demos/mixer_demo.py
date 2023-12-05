import numpy as np
import matplotlib.pyplot as plt
from utils import create_signal, plot_spec, create_phase_modulated


def main():
    n = 1000
    time = np.linspace(0, 2.0, n)
    fs = n / time[1]
    sig1 = create_signal(1.0, 10000, 1.5 * np.pi, time)
    sig2 = create_signal(10.0, 50000, 0, time)
    sig12 = sig1 * sig2

    plt.subplot(411)
    plt.plot(time, sig1)
    plt.plot(time, sig2)

    plt.subplot(412)
    plt.plot(time, sig1 * sig2)

    plt.subplot(413)
    plot_spec(sig1, fs)
    plot_spec(sig2, fs)

    plt.subplot(414)
    plot_spec(sig12, fs)
    plt.show()


if __name__ == '__main__':
    main()
