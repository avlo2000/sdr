import numpy as np
from matplotlib import pyplot as plt


class Despiker:
    def __init__(self, data: np.ndarray):
        self.data = data

    def despike(self):
        threshold = np.pi / 7
        data_despiked = self.data.copy()
        spice_indices = []
        for row in data_despiked:
            spike_idx = []
            for i in range(len(row) - 2):
                d01 = row[i + 1] - row[i]
                d12 = row[i + 2] - row[i + 1]
                ang = np.arctan2(1.0, abs(d01)) + np.arctan2(1.0, abs(d12))
                if d01 * d12 < 0 and (ang < threshold):
                    spike_idx.append(i + 1)

            spice_indices.append(spike_idx)
        filtered_dub_spikes = [[]] * len(spice_indices)
        for i, spikes in enumerate(spice_indices):
            for j in range(len(spikes) - 1):
                if spikes[j] != spikes[j + 1]:
                    filtered_dub_spikes[i].append(spikes[j])
        for i, row_spikes in enumerate(filtered_dub_spikes):
            for j in row_spikes:
                data_despiked[i, j] = 0.5 * (data_despiked[i, j - 1] + data_despiked[i, j + 1])

        return data_despiked
