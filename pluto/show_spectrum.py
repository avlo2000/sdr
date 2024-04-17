import adi
import numpy as np
import time
import cv2

sample_rate = 30e6  # Hz
center_freq = int(2.4e9)  # Hz
num_samps = 1000  # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'fast_attack'
sdr.rx_hardwaregain_chan0 = 50.0  # dB, increase to increase the receive gain, but be careful not to saturate the ADC

for _ in range(20):
    _ = sdr.rx()

samples_cnt = 500
spec_view = np.zeros([samples_cnt, num_samps])
spec_idx = 0
while True:
    t0 = time.time()
    rx_data = sdr.rx()
    print(f"Time: {time.time() - t0}")

    psd = np.abs(np.fft.fftshift(np.fft.fft(rx_data))) ** 2
    psd_dB = 10 * np.log10(psd)
    spec_view[spec_idx, :] = psd_dB * 10
    if spec_idx < samples_cnt - 1:
        spec_idx += 1
    else:
        spec_view = np.roll(spec_view, 1, axis=0)
    cv2.imshow("spec_view", spec_view)
    cv2.waitKey(20)

    f = np.linspace(sample_rate / -2, sample_rate / 2, len(psd))

