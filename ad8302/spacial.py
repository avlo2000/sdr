import numpy as np
from scipy.constants import c


def ang_diff(ang0: np.ndarray, ang1: np.ndarray) -> np.ndarray:
    diff = ang0 - ang1 - np.pi
    return np.arctan2(np.sin(diff), np.cos(diff))


def calc_phase(src: np.ndarray, ant: np.ndarray, freq: float) -> np.ndarray:
    wavelength = c / freq
    src_inv = src.copy()
    d = np.linalg.norm(src_inv - ant, axis=1)
    phs = (2.0 * np.pi * d / wavelength) % (2 * np.pi) - np.pi
    return phs


def calc_phase_diff(src_loc: np.ndarray, ants_loc: np.ndarray, topology: np.ndarray, freq: float) -> np.ndarray:
    phases = calc_phase(src_loc[0], ants_loc, freq)
    d_phases = np.array([ang_diff(phases[i], phases[j]) for (i, j) in topology])
    return d_phases
