import dataclasses
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from ad8302.analysis.despike import Despiker
from ad8302.analysis.filter import HistFilter


@dataclasses.dataclass(init=False)
class Sample:
    src_loc: np.ndarray
    phs_data: np.ndarray


def load_sample(sample_dict: Dict) -> Sample:
    sample = Sample()
    sample.src_loc = np.array(sample_dict['src_loc'])
    phs_data = []
    for i in range(len(sample_dict) - 1):
        phs_data.append(sample_dict[f'phs_data_{i}'])
    sample.phs_data = np.array(phs_data)
    return sample


def load_data(path_to_json: Path):
    with open(path_to_json, 'r') as fp:
        data = json.load(fp)
    for sample_dict in data:
        yield load_sample(sample_dict)


if __name__ == '__main__':
    def main():
        res = list(load_data(Path('../data/dataset3m_home_31_07.json')))
        raw_data = [[], [], [], []]
        for sample in res:
            for i, phs in enumerate(sample.phs_data):
                raw_data[i].extend(phs)

        raw_data = np.array(raw_data)

        plt.subplot(311)
        for data in raw_data:
            plt.plot(data)
        data_despiked = raw_data.copy()
        for _ in range(100):
            data_despiked = Despiker(data_despiked).despike()
        plt.subplot(312)
        for data in data_despiked:
            plt.plot(data)

        plt.subplot(313)
        plt.scatter(np.sin(np.deg2rad(data_despiked[0])), np.deg2rad(data_despiked[1]))

        plt.show()
        print(np.max(raw_data, axis=1) - np.min(raw_data, axis=1))
    main()
