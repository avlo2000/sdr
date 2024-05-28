import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c


def ang_diff(ang0: np.ndarray, ang1: np.ndarray):
    return np.arctan2(np.sin(ang0 - ang1), np.cos(ang0 - ang1))


class Beamformer:
    def __init__(self, freq: float, d_to_ref: np.ndarray):
        self.d_to_ref = d_to_ref
        self.wavelength = c / freq
        print(f"wavelength {self.wavelength}")

    def doa_pattern(self, phases: np.ndarray) -> (np.ndarray, np.ndarray):
        doas = np.linspace(-np.pi, +np.pi, 10000)
        errors = np.empty_like(doas)
        for i, doa in enumerate(doas):
            phases_exp = 2.0 * np.pi * self.d_to_ref * np.sin(doa) / self.wavelength
            error = np.linalg.norm(ang_diff(phases_exp, phases))
            errors[i] = error
        return doas, errors


if __name__ == '__main__':
    def get_phase(src: np.ndarray, ant: np.ndarray, freq: float):
        d = np.linalg.norm(src - ant)
        w = 2.0 * np.pi * freq
        t = d / c
        phs = np.arcsin(np.sin(w * t))
        return phs

    def main():
        freq = 0.433e+9
        ref_loc = np.array([[0.0, 0.0]])
        src_loc = np.array([[-100.0, 3000.0]])
        doa = np.rad2deg(np.arctan2(src_loc[:, 1], src_loc[:, 0]))
        print(f"doa: {doa}")
        ants_loc = np.array(
            [[0.2, 0.0],
             [0.1, 0.0],
             [-0.1, 0.0],
             [-0.2, 0.0]]
        )
        beamformer = Beamformer(freq, (ants_loc - ref_loc)[:, 0])
        plt.subplot(211)
        plt.scatter(src_loc[:, 0], src_loc[:, 1], marker='X')
        plt.scatter(ants_loc[:, 0], ants_loc[:, 1], marker='*')

        phase_ref = get_phase(src_loc[0], ref_loc[0], freq)
        d_phases = np.empty(len(ants_loc))
        for i, loc in enumerate(ants_loc):
            phase = get_phase(src_loc[0], loc, freq)
            d_phases[i] = ang_diff(phase_ref, phase)
        doas, errors = beamformer.doa_pattern(d_phases)

        plt.subplot(212, projection='polar')
        plt.plot(doas, errors)
        plt.show()

    main()
