import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from ad8302.beamformer import Beamformer
from ad8302.spacial import calc_phase_diff


def main():
    freq = 0.433e+9
    ref_loc = np.array([[0.0, 0.0]])
    src_loc = np.array([[130.0, 100.0]])
    ants_loc = np.array(
        [[0.2, 0.0],
         [0.1, 0.0],
         [-0.1, 0.0],
         [-0.2, 0.0]]
    ) * 0.2
    fig = plt.figure()
    x_coord_axis = fig.add_axes((0.25, 0.0, 0.65, 0.03))
    src_loc_x_slider = Slider(
        ax=x_coord_axis,
        label="src x coord",
        valmin=-300,
        valmax=+300,
        valinit=0.0,
        orientation="horizontal"
    )

    beamformer = Beamformer(freq, (ants_loc - ref_loc)[:, 0])
    ax1 = plt.subplot(411)
    geom_loc_scatter, = plt.plot(src_loc[:, 0], src_loc[:, 1], marker='X')
    plt.scatter(ants_loc[:, 0], ants_loc[:, 1], marker='*')
    ax1.set_xlim([-300, +300])

    ax2 = plt.subplot(412)
    doa2err, = plt.plot([], [])
    ax2.set_xlim([-180, +180])
    ax2.set_ylim([0, 1])

    plt.subplot(413, projection='polar')
    doa2err_polar, = plt.plot([], [])

    ax4 = plt.subplot(414)
    phase_data = [[0]] * len(ants_loc)
    phase_lines = []
    time_data = [0]
    for i in range(len(ants_loc)):
        line, = ax4.plot(time_data, phase_data[i])
        phase_lines.append(line)

    def update(_):
        src_loc[:, 0] = src_loc_x_slider.val
        doa = np.rad2deg(np.arctan2(src_loc[:, 1], src_loc[:, 0]))
        print(f"doa real: {doa.item()}")
        geom_loc_scatter.set_xdata(src_loc[:, 0])

        time_data.append(time_data[-1] + 1)
        d_phases = calc_phase_diff(src_loc, ants_loc, ref_loc, freq)
        print(np.rad2deg(d_phases))
        doas, errors = beamformer.doa_pattern(d_phases)
        doa_est = np.rad2deg(doas[np.argmin(errors[np.abs(errors < 100)])]) + 90

        print(f"doa estimated: {doa_est}")
        print()
        errors /= np.max(errors)
        doa2err.set_xdata(np.rad2deg(doas))
        doa2err.set_ydata(errors)

        doa2err_polar.set_xdata(doas)
        doa2err_polar.set_ydata(errors)
        fig.canvas.draw_idle()

    src_loc_x_slider.on_changed(update)

    plt.show()


if __name__ == '__main__':
    main()
