import numpy as np
from scipy.constants import c


def create_linear_array(freq: float):
    wavelen = c / freq
    print(f"Wavelen: {wavelen}")
    topology = np.array([
        [0, 2],
        [1, 2],
        [3, 2],
        [4, 2],
    ])

    ants_loc = []
    for i in range(5):
        ants_loc.append([i * wavelen / 2, 0.0])
    ants_loc = np.array(ants_loc)

    # ants_loc = np.array([
    #     [-0.2, 0.0],
    #     [-0.1, 0.0],
    #     [0.0, 0.0],
    #     [0.1, 0.0],
    #     [0.2, 0.0],
    # ])
    return ants_loc, topology


def create_square_array(freq: float):
    wavelen = c / freq
    print(f"Wavelen: {wavelen}")
    topology = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ])

    ants_loc = np.array([
        [0.0, 0.0],
        [wavelen / 4, 0.0],
        [wavelen / 4, wavelen / 4],
        [0.0, wavelen / 4],
    ])
    return ants_loc, topology


def create_triangular_array(freq: float):
    wavelen = c / freq
    print(f"Wavelen: {wavelen}")
    topology = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
    ])

    ants_loc = np.array([
        [-wavelen / 5, 0.0],
        [0.0, wavelen / 5],
        [+wavelen / 5, 0.0],
    ])
    return ants_loc, topology


