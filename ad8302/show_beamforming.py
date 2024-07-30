from collections import deque
from time import sleep

import numpy as np
from matplotlib import pyplot as plt

from device import Device, v_to_phs, ParseResult, v_to_mag
from beamformer import Beamformer


def main():
    freq = 0.433e9
    d_to_ref = np.array([-0.2, -0.1, 0.1, 0.2])
    beamformer = Beamformer(freq, d_to_ref)
    dev = Device('/dev/ttyACM0')

    fig = plt.figure()
    plt.grid(True)

    num_ant = 4

    ax1 = plt.subplot(411, projection='polar')
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(55)
    ax1.set_xlim([-180, +180])
    ax1.set_ylim([0, 1.1])
    beamloss_polar, = ax1.plot([], [], lw=3, label='beamloss')
    mins_polar, = ax1.plot([], [], lw=3, label='mins')

    ax2 = plt.subplot(412)
    # ax1.set_theta_zero_location('N')
    # ax1.set_theta_direction(-1)
    # ax1.set_rlabel_position(55)
    ax2.set_xlim([-180, +180])
    ax2.set_ylim([0, 1.1])
    beamloss, = ax2.plot([], [], lw=3, label='beamloss')
    mins, = ax2.plot([], [], lw=3, label='mins', marker='o')

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
        print(phs_sym)
        # calib_phase = np.array([-23.81806847, -38.0616589,  -39.5434629,  25.38799953])
        # phases += calib_phase
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
        mins.set_xdata([mns_x, mns_x])
        mins.set_ydata([mns_y, mns_y])
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
