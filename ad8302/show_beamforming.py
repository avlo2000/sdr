from collections import deque
from time import sleep

import numpy as np
from matplotlib import pyplot as plt

from device import Device, v_to_phs, ParseResult, v_to_mag
from beamformer import Beamformer


def main():
    freq = 0.450e9
    d_to_ref = np.array([-0.205, -0.104, 0.100, 0.202])
    beamformer = Beamformer(freq, d_to_ref)
    dev = Device('COM5')

    fig = plt.figure()
    plt.grid(True)

    num_ant = 4
    ax1 = plt.subplot(311)
    # ax1.set_theta_zero_location('N')
    # ax1.set_theta_direction(-1)
    # ax1.set_rlabel_position(55)
    ax1.set_xlim([-180, +180])
    ax1.set_ylim([0, 1.1])
    beamloss, = ax1.plot([], [], lw=3, label='beamloss')
    mins, = ax1.plot([], [], lw=3, label='mins', marker='o')

    ax2 = plt.subplot(312)
    ax2.set_ylim([0.0, 180.0])
    phs_lines = []
    for i in range(num_ant):
        line, = ax2.plot([], lw=3, label=f'ant{i}')
        phs_lines.append(line)

    ax3 = plt.subplot(313)
    ax3.set_ylim([-35, +35])
    mag_lines = []
    for i in range(num_ant):
        line, = ax3.plot([], lw=3, label=f'ant{i}')
        mag_lines.append(line)

    plt.legend()
    fig.canvas.draw()

    plt.show(block=False)

    time_data = deque(maxlen=2000)
    phase_data = deque(maxlen=2000)
    mag_data = deque(maxlen=2000)

    def live_update(t: float, parse: ParseResult):
        phases = v_to_phs(parse.vphs)
        doas, errors = beamformer.doa_pattern(np.deg2rad(phases))
        errors /= np.max(errors)
        doas = np.rad2deg(doas)
        beamloss.set_data(doas, errors)

        idx1, idx2 = np.argsort(errors)[:2]
        if -90 < doas[idx1] < +90:
            idx = idx1
        else:
            idx = idx2
        mns_x = doas[idx]
        mns_y = errors[idx]
        mins.set_xdata(mns_x)
        mins.set_ydata(mns_y)

        plt.grid(True)
        time_data.append(t)
        ax2.set_xlim(max(time_data) - 10.0, max(time_data) + 1.0)
        ax3.set_xlim(max(time_data) - 10.0, max(time_data) + 1.0)

        phase_data.append(phases)
        mag_data.append(v_to_mag(parse.vmag))

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
