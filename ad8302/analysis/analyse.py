from scipy.stats import pearsonr
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from ad8302.analysis.despike import Despiker
from ad8302.analysis.load_data import load_data


if __name__ == '__main__':
    def main():
        res = list(load_data(Path('../data/non_labeled_3m_home_31_07.json')))
        all_phs_data = [[], [], [], []]
        for sample in res:
            for i, phs in enumerate(sample.phs_data):
                all_phs_data[i].extend(phs)

        all_phs_data = np.array(all_phs_data)

        plt.subplot(211)
        for data in all_phs_data:
            plt.plot(data)
        data_despiked = all_phs_data.copy()
        data_despiked = Despiker(data_despiked).despike()
        corr_coef, p_value = pearsonr(data_despiked[0], data_despiked[1])
        print("Pearson correlation coefficient:", corr_coef)
        print("p-value:", p_value)

        corr_coef, p_value = pearsonr(data_despiked[1], data_despiked[2])
        print("Pearson correlation coefficient:", corr_coef)
        print("p-value:", p_value)

        corr_coef, p_value = pearsonr(data_despiked[2], data_despiked[3])
        print("Pearson correlation coefficient:", corr_coef)
        print("p-value:", p_value)

        corr_coef, p_value = pearsonr(data_despiked[0], data_despiked[3])
        print("Pearson correlation coefficient:", corr_coef)
        print("p-value:", p_value)
    main()

