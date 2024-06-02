import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.constants import c

from ad8302.beamformer import Beamformer, ang_diff


def get_phase(src: np.ndarray, ant: np.ndarray, freq: float):
    wavelength = c / freq
    src_inv = src.copy()
    # src_inv[0] *= -1
    d = np.linalg.norm(src_inv - ant)
    phs = 2.0 * np.pi * d / wavelength
    return phs


def main():
    freq = 0.433e+9
    ref_loc = np.array([[0.0, 0.0]])
    src_loc = np.array([[130.0, 100.0]])
    ants_loc = np.array(
        [[0.2, 0.0],
         [0.1, 0.0],
         [-0.1, 0.0],
         [-0.2, 0.0]]
    )
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
    ax1 = plt.subplot(311)
    geom_loc_scatter, = plt.plot(src_loc[:, 0], src_loc[:, 1], marker='X')
    plt.scatter(ants_loc[:, 0], ants_loc[:, 1], marker='*')
    ax1.set_xlim([-300, +300])

    ax2 = plt.subplot(312)
    doa2err, = plt.plot([], [])
    ax2.set_xlim([-180, +180])
    ax2.set_ylim([0, 1])

    plt.subplot(313, projection='polar')
    doa2err_polar, = plt.plot([], [])

    def update(_):
        src_loc[:, 0] = src_loc_x_slider.val
        doa = np.rad2deg(np.arctan2(src_loc[:, 1], src_loc[:, 0]))
        print(f"doa real: {doa.item()}")
        geom_loc_scatter.set_xdata(src_loc[:, 0])
        phase_ref = get_phase(src_loc[0], ref_loc[0], freq)
        d_phases = np.empty(len(ants_loc))
        for i, loc in enumerate(ants_loc):
            phase = get_phase(src_loc[0], np.array(loc), freq)
            d_phases[i] = ang_diff(phase, phase_ref)
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

    src_loc_x_slider.on_changed(update)

    plt.show()


if __name__ == '__main__':
    main()
