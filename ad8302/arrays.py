import numpy as np
from scipy.constants import c


def create_linear_array(freq: float):
    wavelen = c / freq
    topology = np.array([
        [0, 2],
        [1, 2],
        [3, 2],
        [4, 2],
    ])

    ants_loc = []
    for i in range(15):
        ants_loc.append([i * wavelen / 3, 0.0])
    ants_loc = np.array(ants_loc)

    return ants_loc, topology


def create_grably_array(_: float):
    topology = np.array([
        [0, 2],
        [1, 2],
        [3, 2],
        [4, 2],
    ])

    ants_loc = np.array([
        [-0.2, 0.0],
        [-0.1, 0.0],
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
    ])
    return ants_loc, topology


def create_square_array(freq: float):
    wavelen = c / freq
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


def create_circular_array(freq: float):
    wavelen = c / freq
    n = 7
    r = wavelen / (4 * np.sin(np.pi / n))
    topology = []
    ants_loc = []
    for i in range(n):
        topology.append([i, (i + 1) % n])
        ants_loc.append([r * np.cos(2.0 * np.pi * i / n), r * np.sin(2.0 * np.pi * i / n)])
    topology = np.array(topology)
    ants_loc = np.array(ants_loc)
    return ants_loc, topology


def create_circular_one_ref_array(freq: float):
    wavelen = c / freq
    n = 5
    r = wavelen / 3
    topology = []
    ants_loc = []
    for i in range(n):
        if i != n // 2:
            topology.append([i, n // 2])
        ants_loc.append([r * np.cos(2.0 * np.pi * i / n), r * np.sin(2.0 * np.pi * i / n)])
    topology = np.array(topology)
    ants_loc = np.array(ants_loc)
    return ants_loc, topology
