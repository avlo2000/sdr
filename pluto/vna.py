import math

import adi
import numpy as np
from matplotlib import pyplot as plt

from take_sample import take_sample

sdr = adi.Pluto('ip:192.168.2.1')

sdr.rx_rf_bandwidth = 2000000
sdr.sample_rate = 6000000
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = -5
sdr.rx_hardwaregain_chan0 = 30
sdr.gain_control_mode_chan0 = "manual"

sdr.rx_enabled_channels = [0]
sdr.tx_enabled_channels = [0]

freq_range = np.linspace(2.0, 3.0, num=200) * 1e9

gains = np.empty_like(freq_range)

plt.ion()
figure, ax = plt.subplots()

for i, freq in enumerate(freq_range):
    tx, rx, time = take_sample(sdr, int(freq))
    tx_energy = np.trapz(tx, time)
    rx_energy = np.trapz(rx, time)
    fft = np.fft.fft(rx)
    gain = 10 * math.log10(abs(rx_energy) / abs(tx_energy))
    gains[i] = gain

plt.plot(freq_range, gains)
plt.show()
