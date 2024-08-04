import numpy as np
import torch
from scipy.constants import c


def freq_to_wavelength(freq: float):
    return c / freq


def ang_diff_np(ang0: np.ndarray, ang1: np.ndarray) -> np.ndarray:
    diff = ang1 - ang0
    return (diff + np.pi) % (2.0 * np.pi) - np.pi


def calc_phase_np(src: np.ndarray, ant: np.ndarray, freq: float) -> np.ndarray:
    wavelength = freq_to_wavelength(freq)
    src_inv = src.copy()
    d = np.linalg.norm(src_inv - ant, axis=1)
    phs = (2.0 * np.pi * d / wavelength) % (2 * np.pi) - np.pi
    return phs


def calc_phase_diff_np(src_loc: np.ndarray, ants_loc: np.ndarray, topology: np.ndarray, freq: float) -> np.ndarray:
    phases = calc_phase_np(src_loc[0], ants_loc, freq)
    d_phases = np.array([ang_diff_np(phases[i], phases[j]) for (i, j) in topology])
    return d_phases


def ang_diff(ang0: torch.Tensor, ang1: torch.Tensor) -> torch.Tensor:
    diff = ang1 - ang0
    return (diff + torch.pi) % (2.0 * torch.pi) - torch.pi


def calc_phase(src: torch.Tensor, ant: torch.Tensor, freq: float) -> torch.Tensor:
    wavelength = freq_to_wavelength(freq)
    src_inv = src.clone()
    d = torch.linalg.norm(src_inv - ant, axis=1)
    phs = (2.0 * torch.pi * d / wavelength) % (2 * torch.pi) - torch.pi
    return phs


def calc_phase_diff(src_loc: torch.Tensor, ants_loc: torch.Tensor, topology: torch.Tensor, freq: float) -> torch.Tensor:
    phases = calc_phase(src_loc[0], ants_loc, freq)
    d_phases = torch.tensor([ang_diff(phases[i], phases[j]) for (i, j) in topology])
    return d_phases

