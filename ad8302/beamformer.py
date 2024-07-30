import numpy as np
from scipy.constants import c

from ad8302.spacial import ang_diff


# class Beamformer:
#     def __init__(self, freq: float, d_to_ref: np.ndarray):
#         self.d_to_ref = d_to_ref
#         self.wavelength = c / freq
#         print(f"wavelength {self.wavelength}")
#
#     def doa_pattern(self, phases: np.ndarray) -> (np.ndarray, np.ndarray):
#         doas = np.linspace(-np.pi, +np.pi, 1000)
#         errors = np.empty_like(doas)
#         for i, doa in enumerate(doas):
#             phases_exp = 2.0 * np.pi * self.d_to_ref * np.sin(doa) / self.wavelength
#             error = np.linalg.norm(ang_diff(phases_exp, phases))
#             errors[i] = error
#         return doas, errors

class Beamformer:
    def __init__(self, freq: float, ant_locs: np.ndarray, topology: np.ndarray):
        self.ant_locs = ant_locs
        self.topology = topology
        self.wavelength = c / freq

        dists = []
        angles = []
        for (i, j) in topology:
            vec = ant_locs[i] - ant_locs[j]
            dists.append(np.sqrt(vec[0] ** 2 + vec[1] ** 2))
            angles.append(np.arctan2(vec[1], vec[0]))
        self.dists = np.array(dists)
        self.angles = np.array(angles)

    def doa_pattern(self, phases: np.ndarray) -> (np.ndarray, np.ndarray):
        doas = np.linspace(-np.pi, +np.pi, 1000)
        errors = np.empty_like(doas)

        dbg = []
        for i, doa in enumerate(doas):
            phases_expected = 2.0 * np.pi * self.dists * np.sin(doa + self.angles) / self.wavelength
            dbg.append(phases_expected[0])
            psi = abs(phases_expected) - phases
            error = np.exp(-0.5 * psi.dot(psi))
            errors[i] = error
        print(f"Min: {min(np.rad2deg(dbg))}, Max: {np.rad2deg(max(dbg))}")
        return doas, errors
