import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider


# The parametrized function to be plotted
def f(t, phase, frequency):
    return np.sin(2 * np.pi * frequency * t + phase)


def f_fft(t, phase, frequency):
    fft = np.fft.fft(f(t, phase, frequency))
    frq = np.fft.fftfreq(len(t), t[1])
    return frq, np.real(fft), np.imag(fft)


t = np.linspace(0, 1, 1000)

# Define initial parameters
init_phase = 5
init_frequency = 3

# Create the figure and the line that we will manipulate
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
signal1, = ax1.plot(t, f(t, init_phase, init_frequency), lw=2)
signal2, = ax1.plot(t, f(t, 0.0, init_frequency), lw=2)

signal_sum, = ax2.plot(t, f(t, init_phase, init_frequency) * f(t, 0.0, init_frequency), lw=2)

ax1.set_xlabel('Time [s]')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [Hz]',
    valmin=0.1,
    valmax=400,
    valinit=init_frequency,
)

# Make a vertically oriented slider to control the amplitude
axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
phase_slider = Slider(
    ax=axamp,
    label="Phase",
    valmin=0,
    valmax=360,
    valinit=init_phase,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    y_rf = f(t, np.pi * phase_slider.val / 180, freq_slider.val)
    y_lo = f(t, 0.0, 2)
    signal1.set_ydata(y_rf)
    signal2.set_ydata(y_lo)
    signal_sum.set_ydata(y_rf * y_lo)


freq_slider.on_changed(update)
phase_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    phase_slider.reset()


button.on_clicked(reset)

plt.show()
