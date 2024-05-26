from collections import deque

import numpy as np
from matplotlib import pyplot as plt

from ad8302.device import Device, v_to_phs
from ad8302.beamformer import Beamformer


def main():
    freq = 0.433e9
    d_to_ref = np.array([0.205, 0.104, 0.100, 0.202])
    beamformer = Beamformer(freq, d_to_ref)
    dev = Device('COM5')

    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(55)
    beamloss, = ax1.plot([], [], lw=3)
    fig1.canvas.draw()

    fig2, ax2 = plt.subplots()
    ax2.set_ylim([0.0, -180.0])
    plt.grid(True)

    num_ant = 4
    phs_lines = []
    for _ in range(num_ant):
        line, = ax2.plot([], lw=3)
        phs_lines.append(line)
    fig2.canvas.draw()

    plt.show(block=False)

    x_data = deque(maxlen=2000)
    y_data = deque(maxlen=2000)

    def live_update(t: float, volt: np.ndarray):
        phases = v_to_phs(volt)
        doas, errors = beamformer.doa_pattern(phases)
        errors /= np.max(errors)
        beamloss.set_data(doas, errors)
        print(f"Time: {t}")
        fig1.canvas.draw()
        fig1.canvas.flush_events()

        x_data.append(t)
        y_data.append(phases)
        ys = np.array(y_data)

        for i, y in enumerate(ys.T):
            phs_lines[i].set_data(x_data, y)
        ax2.set_xlim(max(x_data) - 10.0, max(x_data))
        fig2.canvas.draw()
        fig2.canvas.flush_events()

    dev.spin(live_update)


if __name__ == '__main__':
    main()
