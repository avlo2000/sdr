import adi
import matplotlib.pyplot as plt
import numpy as np
import time

from take_sample import take_sample, create_rnd_signal

sample_rate = 1.92e6  # Hz
center_freq = int(2.4e9)  # Hz
num_samps = 100000  # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 10.0  # dB, increase to increase the receive gain, but be careful not to saturate the ADC

# Start the transmitter
sdr.tx_cyclic_buffer = True  # Enable cyclic buffers
for _ in range(20):  # for Pluto initialization
    rx_data = sdr.rx()

plt.ion()
figure, ax = plt.subplots()
rx_spec_plot = None

ax.set_xlim(sample_rate / -2, sample_rate / 2)
ax.set_ylim(0, 100)

while True:
    t0 = time.time()
    rx_data = sdr.rx()
    print(f"Time: {time.time() - t0}")

    psd = np.abs(np.fft.fftshift(np.fft.fft(rx_data))) ** 2
    psd_dB = 10 * np.log10(psd)

    f = np.linspace(sample_rate / -2, sample_rate / 2, len(psd))
    if rx_spec_plot is None:
        rx_spec_plot,  = ax.plot(f, rx_data)
    rx_spec_plot.set_xdata(f)
    rx_spec_plot.set_ydata(psd_dB)

    figure.canvas.draw()

    figure.canvas.flush_events()
    time.sleep(0.001)
