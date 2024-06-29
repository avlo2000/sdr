import matplotlib.pyplot as plt
import numpy as np


class PhaseSpace2:
    def __init__(self, n: float, m: float):
        self._n = n
        self._m = m

    def plot(self, ax: plt.Axes, n_lines: int, n2: int):
        ax.set_aspect('equal')
        k = self._n / self._m

        n1_r = int((self._m + 1) // 2)
        n2_r = int((self._m + 1) // 2)
        for n1 in range(-n1_r, +n1_r):
            for n2 in range(-n2_r, +n2_r):
                b = 2.0 * np.pi * (self._n * n1 - self._m * n2) / self._m
                xp0 = -np.pi
                xp1 = +np.pi
                yp0 = k * xp0 + b
                yp1 = k * xp1 + b
                x = np.rad2deg([xp0, xp1])
                y = np.rad2deg([yp0, yp1])
                ax.plot(y, x, c='blue')


class PhaseSpace:
    def __init__(self, lambdas_to_ref: np.ndarray, freq: np.ndarray):
        pass


if __name__ == '__main__':
    def main():
        ps2 = PhaseSpace2(5, 8)
        ax = plt.subplot(111)
        ps2.plot(ax, 50, 1)
        plt.show()
    main()
