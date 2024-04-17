import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import load_dump


def main():
    central_f = 1.57542e9  # L1
    iq = load_dump.read_int16_to_iq(Path('data/out4.int16'))
    sample_rate = 40e6
    psd = np.abs(np.fft.fftshift(np.fft.fft(iq))) ** 2
    psd_dB = 10 * np.log10(psd)

    f = np.linspace(sample_rate / -2, sample_rate / 2, len(psd))

    plt.plot(f, psd_dB)
    plt.show()


if __name__ == '__main__':
    main()
