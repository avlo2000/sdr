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
