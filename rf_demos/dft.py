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
signal, = ax1.plot(t, f(t, init_phase, init_frequency), lw=2)

frq, real, imag = f_fft(t, init_phase, init_frequency)
fft_re, = ax2.plot(frq, real, lw=2)
fft_im, = ax2.plot(frq, imag, lw=2)
fft_sum, = ax3.plot(frq, abs(real) + abs(imag), lw=2)

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
    signal.set_ydata(f(t, np.pi * phase_slider.val / 180, freq_slider.val))
    frq, real, imag = f_fft(t, np.pi * phase_slider.val / 180, freq_slider.val)
    fft_re.set_ydata(real)
    fft_im.set_ydata(imag)
    ax2.set_ylim(min(min(real), min(imag)), max(max(real), max(imag)))
    fft_abs = real**2 + imag**2
    fft_sum.set_ydata(fft_abs)
    ax3.set_ylim(min(fft_abs) - 500, max(fft_abs) + 500)
    fig.canvas.draw_idle()


# register the update function with each slider
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