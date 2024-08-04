import matplotlib.pyplot as plt
import numpy as np

from ad8302.spacial import ang_diff_np

if __name__ == '__main__':
    phases = np.linspace(-np.pi, np.pi, 200)
    ref_phs = np.zeros(1)
    res = np.empty_like(phases)
    for i, phs in enumerate(phases):
        res[i] = abs(ang_diff_np(phs + np.pi, ref_phs))
    plt.plot(np.rad2deg(phases), np.rad2deg(res))
    plt.show()
