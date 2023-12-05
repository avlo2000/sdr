import numpy as np
from matplotlib import pyplot as plt

from rf_demos.utils import create_random_iq_ref, create_signal

iq = create_random_iq_ref(10)
n = 10000
time = np.linspace(0, 1.0, n)
fs = n / time[1]

iq = np.repeat(iq, n // len(iq))[:n]
carrier_q = create_signal(0.1, 500000, np.pi / 2, time)
carrier_i = create_signal(0.1, 500000, 0, time)

air = carrier_q * np.real(iq) + carrier_i * np.imag(iq)

plt.subplot(411)
plt.plot(time, np.real(iq))
plt.plot(time, np.imag(iq))
plt.subplot(412)
plt.plot(time, air)
plt.show()
