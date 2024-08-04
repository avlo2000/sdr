import numpy as np
from scipy.constants import c


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

    def doa_to_phases(self, doa: np.ndarray):
        r_doa = doa[np.newaxis].repeat(len(self.angles), 0) + self.angles[:, np.newaxis]
        return np.abs(2.0 * np.pi * self.dists[:, np.newaxis] * np.cos(r_doa) / self.wavelength)

    def doa_pattern(self, phases: np.ndarray, resolution: int) -> (np.ndarray, np.ndarray):
        doas = np.linspace(-np.pi, +np.pi, resolution)

        phases_expected = self.doa_to_phases(doas)
        psi = phases_expected - phases[:, np.newaxis]
        errors = np.exp(-0.5 * np.sum(psi * psi, axis=0))
        return doas, errors

