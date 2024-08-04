import matplotlib.pyplot as plt
import numpy as np
import torch

from ad8302.arrays import create_grably_array
from ad8302.beamformer import Beamformer

"""Goal of my life is to map measured phases onto this function"""


class PhaseSpace:
    def __init__(self, freq: float, ants_loc: np.ndarray, topology: np.ndarray):
        self.freq = freq
        self.ants_loc = ants_loc
        self.topology = topology
        self.beamformer = Beamformer(freq, ants_loc, topology)

    def phase_curve(self, resolution: int):
        doas = np.linspace(-np.pi, +np.pi, resolution)
        phases = self.beamformer.doa_to_phases(doas)
        return doas, phases

    def plot(self, resolution: int, ax: plt.Axes) -> (np.ndarray, np.ndarray):
        doas, phases = self.phase_curve(resolution)
        for i, phase in enumerate(phases):
            ax.plot(doas, phase, label=f'phase{i}')


if __name__ == '__main__':
    def main():
        freq = 0.433e+9
        ants_loc, topology = create_grably_array(freq)
        ps2 = PhaseSpace(freq, ants_loc, topology)
        ax = plt.subplot(111)
        ps2.plot(1000, ax)
        plt.legend()
        plt.show()


    main()
