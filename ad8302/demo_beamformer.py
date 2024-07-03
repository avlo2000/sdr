from collections import deque

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from ad8302.beamformer import Beamformer
from ad8302.spacial import calc_phase_diff


def main():
    freq = 0.433e+9
    ref_loc = np.array([[0.0, 0.0]])
    src_r = 3.0

    ants_loc = np.array(
        [[0.3, 0.0],
         [0.7, 0.0],
         [-0.1, 0.0],
         [-0.2, 0.0]]
    )
    fig = plt.figure()
    x_coord_axis = fig.add_axes((0.25, 0.0, 0.65, 0.03))
    src_loc_ang_slider = Slider(
        ax=x_coord_axis,
        label="src x coord",
        valmin=-180,
        valmax=+180,
        valinit=0.0,
        orientation="horizontal"
    )

    beamformer = Beamformer(freq, (ants_loc - ref_loc)[:, 0])
    ax1 = plt.subplot(511)

    ang = np.deg2rad(src_loc_ang_slider.val)
    src_loc = np.array([[np.cos(ang) * src_r, np.sin(ang) * src_r]])
    geom_loc_scatter, = plt.plot(src_loc[:, 0], src_loc[:, 1], marker='X')
    plt.scatter(ants_loc[:, 0], ants_loc[:, 1], marker='*')
    ax1.set_xlim([-src_r, +src_r])
    ax1.set_ylim([-src_r, +src_r])

    ax2 = plt.subplot(512)
    doa2err, = plt.plot([], [])
    ax2.set_xlim([-240, +240])
    ax2.set_ylim([0, 1])

    plt.subplot(413, projection='polar')
    doa2err_polar, = plt.plot([], [])

    ax4 = plt.subplot(514)
    phs_lines = []
    n = 200
    phase_data = deque(maxlen=n)
    ax4.set_xlim([0, phase_data.maxlen])
    ax4.set_ylim([-240, +240])
    for i in range(len(ants_loc)):
        line, = ax4.plot([], [])
        phs_lines.append(line)

    ax5 = plt.subplot(515)
    ax5.set_xlim([-240, +240])
    ax5.set_ylim([-240, +240])
    scatter, = ax5.plot([], [], marker='o')
    ax5.set_aspect('equal')

    def update(_):
        ang = np.deg2rad(src_loc_ang_slider.val)
        src_loc = np.array([[np.cos(ang) * src_r, np.sin(ang) * src_r]])
        doa = np.rad2deg(np.arctan2(src_loc[:, 1], src_loc[:, 0]))
        print(f"doa real: {doa.item()}")
        geom_loc_scatter.set_xdata(src_loc[:, 0])
        geom_loc_scatter.set_ydata(src_loc[:, 1])

        d_phases = calc_phase_diff(src_loc, ants_loc, ref_loc, freq)

        # Making it more realistic
        # d_phases += np.deg2rad(np.array([38, 40, -11, 65]))
        # d_phases[d_phases >= np.pi / 2] = np.pi - d_phases[d_phases >= np.pi / 2]
        # d_phases[d_phases <= -np.pi / 2] = np.pi + d_phases[d_phases <= -np.pi / 2]

        phase_data.append(np.rad2deg(d_phases))
        for j, y in enumerate(np.array(phase_data).T):
            phs_lines[j].set_data(np.arange(len(y)), y)

        scatter.set_data(np.array(phase_data).T[0], np.array(phase_data).T[1])

        doas, errors = beamformer.doa_pattern(d_phases)
        doa_est = np.rad2deg(doas[np.argmin(errors[np.abs(errors < 100)])]) + 90

        print(f"doa estimated: {doa_est}")
        print()
        errors /= np.max(errors)
        doa2err.set_xdata(-np.rad2deg(doas))
        doa2err.set_ydata(errors)

        doa2err_polar.set_xdata(doas)
        doa2err_polar.set_ydata(errors)
        fig.canvas.draw_idle()

    src_loc_ang_slider.on_changed(update)

    plt.show()


if __name__ == '__main__':
    main()
