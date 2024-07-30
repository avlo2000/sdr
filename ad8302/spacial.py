import numpy as np
from scipy.constants import c


def freq_to_wavelength(freq: float):
    return c / freq


def ang_diff(ang0: np.ndarray, ang1: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(ang0 - ang1), np.cos(ang0 - ang1))


def calc_phase(src: np.ndarray, ant: np.ndarray, freq: float) -> np.ndarray:
    wavelength = freq_to_wavelength(freq)
    src_inv = src.copy()
    d = np.linalg.norm(src_inv - ant)
    phs = 2.0 * np.pi * d / wavelength
    return phs


def calc_phase_diff(src_loc: np.ndarray, ants_loc: np.ndarray, ref_loc: np.ndarray, freq: float) -> np.ndarray:
    phase_ref = calc_phase(src_loc[0], ref_loc[0], freq)
    d_phases = np.empty(len(ants_loc))

    for j, loc in enumerate(ants_loc):
        phase = calc_phase(src_loc[0], np.array(loc), freq)
        # d_phases[j] = ang_diff(phase, phase_ref)
        d_phases[j] = phase - phase_ref

    return d_phases
