from pathlib import Path
import numpy as np
import pandas as pd


def read_int16_to_iq(path: Path) -> np.ndarray:
    data = np.fromfile(path, np.int16)
    i = data[::2]
    q = data[1::2]
    return i + q * 1.0j
