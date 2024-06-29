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
        res = load_data(Path('../data/football_field/dataset_4m.json'))
        for sample in res:
            filt = HistFilter(5)
            for phs in sample.phs_data:
                phs_filtered = filt.filter(phs)
                plt.plot(phs_filtered)
            plt.show()
    main()
