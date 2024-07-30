import dataclasses
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

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
        res = list(load_data(Path('../data/non_labeled.json')))
        all_phs_data = [[], [], [], []]
        for sample in res:
            for i, phs in enumerate(sample.phs_data):
                all_phs_data[i].extend(phs)

        all_phs_data = np.array(all_phs_data)
        plt.plot(all_phs_data[0])
        plt.plot(all_phs_data[1])
        plt.plot(all_phs_data[2])
        plt.plot(all_phs_data[3])
        plt.show()
        print(np.max(all_phs_data, axis=1) - np.min(all_phs_data, axis=1))
    main()
