import math
import time

import adi
import numpy as np
import scipy
import tqdm as tqdm
from matplotlib import pyplot as plt

from take_sample import take_sample, create_rnd_signal


freq_start = int(1.0 * 1e9)
freq_end = int(2.0 * 1e9)
sample_rate = 20e6  # Hz
num_samps = 100000  # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate)  # filter cutoff, just set it to the same as sample rate
sdr.tx_hardwaregain_chan0 = -50  # Increase to increase tx power, valid range is -90 to 0 dB

# Config Rx
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 5.0  # dB, increase to increase the receive gain, but be careful not to saturate the ADC

freq_range = np.linspace(freq_start, freq_end, num=80)
freq_stride = freq_range[1] - freq_range[0]

num_symbols = 1000
samples = create_rnd_signal(num_symbols)

# Start the transmitter
sdr.tx_cyclic_buffer = True  # Enable cyclic buffers
for _ in range(20):  # for Pluto initialization
    take_sample(sdr, int(freq_range[0]), samples)

psd_tx = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2


def perform_measurement():
    whole_n = 900000
    freq_min = freq_start - sample_rate // 2
    freq_max = freq_end + sample_rate // 2
    whole_freq_range = np.linspace(freq_min, freq_max, num=whole_n)
    whole_psd = np.zeros(whole_n)

    for i, center_freq in tqdm.tqdm(enumerate(freq_range)):
        rx_data = take_sample(sdr, int(center_freq), samples)

        psd_rx = np.abs(np.fft.fftshift(np.fft.fft(rx_data))) ** 2
        freq_band = np.linspace(sample_rate / -2, sample_rate / 2, len(psd_rx))
        freq = freq_band + center_freq

        idxs = (whole_n * (freq - freq_min) / (freq_max - freq_min + 1)).astype(int)
        whole_psd[idxs] += psd_rx
    return whole_freq_range, whole_psd


for _ in range(1):
    whole_freq_range, whole_psd = perform_measurement()
    plt.plot(whole_freq_range, whole_psd)
plt.show()
