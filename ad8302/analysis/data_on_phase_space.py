from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ad8302.analysis.load_data import load_data
from ad8302.analysis.phase_space import PhaseSpace2
from ad8302.spacial import freq_to_wavelength
import seaborn as sns


def main():
    ant_locs = np.array([-0.2, -0.1, 0.1, 0.2])
    freq = 433_000_000
    wavelen = freq_to_wavelength(freq)
    print(f'Half wavelength: {wavelen / 2}')

    ant_numbs = ant_locs / (wavelen / 2)
    print(ant_numbs)

    phase_spaces = []
    for i in range(len(ant_numbs) - 1):
        phase_spaces.append(PhaseSpace2(ant_numbs[0], ant_numbs[i + 1]))

    # ax = plt.subplot(111)

    dataset = load_data(Path('../data/football_field/dataset_4m.json'))
    for sample in dataset:
        data = {f"phs{i}": phs_data for i, phs_data in enumerate(sample.phs_data)}

        df = pd.DataFrame(data=data)

        sns.pairplot(df)
        plt.show()


if __name__ == '__main__':
    main()
