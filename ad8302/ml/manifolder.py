from typing import Callable

import numpy as np
import torch
from torch import nn

from ad8302.analysis.phase_space import PhaseSpace
from ad8302.arrays import create_grably_array
from ad8302.spacial import calc_phase_diff_np, freq_to_wavelength, ang_diff
from ad8302.beamformer import Beamformer


class PhaseProjectionModel(nn.Module):
    def __init__(self, n_phases: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Linear(n_phases, n_phases, bias=True)
        # nn.Sequential(
        #     nn.Linear(n_phases, n_phases, bias=True)
            # nn.Tanh(),
            # nn.Linear(16, 16, bias=True),
            # nn.Tanh(),
            # nn.Linear(16, n_phases, bias=True),
            # nn.Tanh(),
        # ))
        self.model.weight.data.copy_(torch.eye(n_phases))

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        prj = self.model(phases)
        return prj


class InterferometryPredictor(nn.Module):
    def __init__(self, freq, ants_loc, topology, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.doa = nn.Parameter(torch.ones(1))
        self.wavelength = freq_to_wavelength(freq)
        beamformer = Beamformer(freq, ants_loc, topology)
        self.dists = torch.tensor(beamformer.dists)
        self.angles = torch.tensor(beamformer.angles)

        mx_phs_val = torch.max(2.0 * torch.pi * self.dists * torch.cos(torch.pi + self.angles) / self.wavelength)
        self.normalizer = mx_phs_val

    def forward(self, phases: torch.Tensor):
        phases = torch.acos(torch.cos(phases))
        doa = torch.acos(phases * self.wavelength / (2.0 * torch.pi * self.dists)) - self.angles
        if torch.any(torch.isnan(doa)):
            return
        return doa


class PhaseManifoldLoss(nn.Module):
    def __init__(self, freq, ants_loc, topology, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wavelength = freq_to_wavelength(freq)
        beamformer = Beamformer(freq, ants_loc, topology)
        self.dists = torch.tensor(beamformer.dists)
        self.angles = torch.tensor(beamformer.angles)

    def doa_to_phases(self, doa: torch.Tensor):
        phases = 2.0 * torch.pi * self.dists * torch.cos(doa + self.angles) / self.wavelength
        return phases

    def forward(self, phases: torch.Tensor, doa: torch.Tensor):
        psi = phases - torch.abs(self.doa_to_phases(doa))
        return torch.exp(-0.5 * psi @ psi.T)


class DOAPatternLoss(nn.Module):
    def __init__(self, freq, ants_loc, topology, resolution: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ps = PhaseSpace(freq, ants_loc, topology)
        _, possible_phases = ps.phase_curve(resolution)
        self.possible_phases = torch.tensor(possible_phases).T

    def forward(self, phases: torch.Tensor):
        similarity = torch.sum((self.possible_phases - phases) ** 2, dim=1)
        return torch.min(similarity)

