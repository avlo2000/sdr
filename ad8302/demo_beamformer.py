from collections import deque

import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from ad8302.arrays import create_grably_array, create_square_array, create_triangular_array, create_circular_array, \
    create_linear_array, create_circular_one_ref_array
from ad8302.beamformer import Beamformer
from ad8302.spacial import calc_phase_diff_np


def main():
    freq = 0.433e+9
    src_r = 300.0

    fig = plt.figure()
    x_coord_axis = fig.add_axes((0.25, 0.0, 0.65, 0.03))
    src_loc_ang_slider = Slider(
        ax=x_coord_axis,
        label="AOA",
        valmin=-180,
        valmax=+180,
        valinit=0.0,
        orientation="horizontal"
    )

    ants_loc, topology = create_grably_array(freq)
    beamformer = Beamformer(freq, ants_loc, topology)

    ax1 = plt.subplot(411)

    ang = np.deg2rad(src_loc_ang_slider.val)
    src_loc = np.array([[np.cos(ang) * src_r, np.sin(ang) * src_r]])
    geom_loc_scatter, = plt.plot(src_loc[:, 0], src_loc[:, 1], marker='X')
    plt.scatter(ants_loc[:, 0], ants_loc[:, 1], marker='*')
    ax1.set_xlim([-src_r, +src_r])
    ax1.set_ylim([-src_r, +src_r])

    ax2 = plt.subplot(412)
    doa2err, = plt.plot([], [])
    ax2.set_xlim([-200, +200])
    ax2.set_ylim([0, 1])

    plt.subplot(413, projection='polar')
    doa2err_polar, = plt.plot([], [])

    ax4 = plt.subplot(414)
    phs_lines = []
    n = 200
    phase_data = deque(maxlen=n)
    ax4.set_xlim([0, phase_data.maxlen])
    ax4.set_ylim([-180, +180])
    for i in range(len(ants_loc)):
        line, = ax4.plot([], [])
        phs_lines.append(line)

    def update(_):
        ang = np.deg2rad(src_loc_ang_slider.val)
        src_loc = np.array([[np.cos(ang) * src_r, np.sin(ang) * src_r]])
        doa = np.rad2deg(np.arctan2(src_loc[:, 1], src_loc[:, 0]))
        print(f"doa real: {doa.item()}")
        geom_loc_scatter.set_xdata(src_loc[:, 0])
        geom_loc_scatter.set_ydata(src_loc[:, 1])

        d_phases = abs(calc_phase_diff_np(src_loc, ants_loc, topology, freq)) + np.random.normal(scale=0.002, size=len(topology))

        phase_data.append(np.rad2deg(d_phases))
        for j, y in enumerate(np.array(phase_data).T):
            phs_lines[j].set_data(np.arange(len(y)), y)

        doas, errors = beamformer.doa_pattern(d_phases, 1000)
        peaks, _ = find_peaks(errors)

        print(f"doa estimated: {np.rad2deg(doas[peaks])}")
        print()

        doa2err.set_xdata(-np.rad2deg(doas))
        doa2err.set_ydata(errors)

        doa2err_polar.set_xdata(doas)
        doa2err_polar.set_ydata(errors)
        fig.canvas.draw_idle()

    src_loc_ang_slider.on_changed(update)

    plt.show()


if __name__ == '__main__':
    main()
