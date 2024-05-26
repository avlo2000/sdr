import time

import adi
import cv2
import numpy as np
from scipy import signal

from take_sample import create_rnd_signal

freq = int(0.433e9)
sample_rate = 1.688e6  # Hz
num_samps = 299999  # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate)  # filter cutoff, just set it to the same as sample rate
sdr.tx_hardwaregain_chan0 = 0.0  # Increase to increase tx power, valid range is -90 to 0 dB
sdr.tx_lo = freq

# Config Rx
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.rx_output_type = 'raw'
sdr.rx_annotated = False
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 50.0
sdr.rx_lo = freq

num_symbols = 100
samples = create_rnd_signal(num_symbols)

sdr.tx_cyclic_buffer = True
sdr.tx(samples)

for i in range(0, 10):
    raw_data = sdr.rx()

while True:
    t0 = time.time()
    x = sdr.rx()
    print(f"Acquisition time: {time.time() - t0}s")

    f, t, Sxx = signal.spectrogram(x, sample_rate, nperseg=1024)
    spec = np.abs(Sxx)
    spec /= np.max(spec)
    sdr.tx_destroy_buffer()
    sdr.tx(samples)

    cv2.imshow('spec', spec.T)
    cv2.waitKey(10)
