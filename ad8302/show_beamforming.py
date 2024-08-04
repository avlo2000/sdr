from collections import deque
from time import sleep

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from ad8302.arrays import create_grably_array
from device import Device, v_to_phs, ParseResult, v_to_mag
from beamformer import Beamformer


def main():
    freq = 0.433e9

    ants_loc, topology = create_grably_array(freq)
    beamformer = Beamformer(freq, ants_loc, topology)

    # dev = Device('COM5')
    dev = Device('/dev/ttyACM0')

    fig = plt.figure()
    plt.grid(True)

    num_ant = 4

    ax1 = plt.subplot(411, projection='polar')
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(55)
    ax1.set_ylim([0, 1.1])
    beamloss_polar, = ax1.plot([], [], lw=3, label='beamloss')

    ax2 = plt.subplot(412)
    ax2.set_xlim([-180, +180])
    ax2.set_ylim([0, 1.1])
    beamloss, = ax2.plot([], [], lw=3, label='beamloss')
    peaks_plot, = ax2.plot([], [], lw=3, label='peaks', marker='o')

    ax3 = plt.subplot(413)
    ax3.set_ylim([-190.0, +190.0])
    phs_lines = []
    for i in range(num_ant):
        line, = ax3.plot([], lw=3, label=f'ant{i}')
        phs_lines.append(line)

    ax4 = plt.subplot(414)
    ax4.set_ylim([-35, +35])
    mag_lines = []
    for i in range(num_ant):
        line, = ax4.plot([], lw=3, label=f'ant{i}')
        mag_lines.append(line)

    plt.legend()
    fig.canvas.draw()

    plt.show(block=False)

    n = 2000
    time_data = deque(maxlen=n)
    phase_data = deque(maxlen=n)
    mag_data = deque(maxlen=n)

    def live_update(t: float, parse: ParseResult):
        phases = v_to_phs(parse.vphs)
        phs_sym = phases[:2] - phases[2:]
        calib_phase = np.array([-55.90190513,   5.00437745, -63.71424255,   1.47771887])
        phases += calib_phase
        doas, errors = beamformer.doa_pattern(np.deg2rad(phases), 1000)
        doas = np.rad2deg(doas)
        beamloss.set_data(doas, errors)

        peaks, info = find_peaks(errors)

        peaks_plot.set_xdata(doas[peaks])
        peaks_plot.set_ydata(errors[peaks])
        beamloss_polar.set_xdata(np.deg2rad(doas))
        beamloss_polar.set_ydata(errors)

        plt.grid(True)
        time_data.append(t)
        ax3.set_xlim(max(time_data) - 10.0, max(time_data) + 1.0)
        ax4.set_xlim(max(time_data) - 10.0, max(time_data) + 1.0)

        phase_data.append(phases)
        mag_data.append(phs_sym)

        for j, y in enumerate(np.array(phase_data).T):
            phs_lines[j].set_data(time_data, y)

        for j, y in enumerate(np.array(mag_data).T):
            mag_lines[j].set_data(time_data, y)

        fig.canvas.draw()
        fig.canvas.flush_events()
        sleep(0.01)

    dev.spin(live_update)


if __name__ == '__main__':
    main()
