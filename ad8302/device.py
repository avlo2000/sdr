import time
from typing import Callable

import numpy as np
import serial


def parse(s: str) -> np.ndarray:
    res = np.array([float(val.split(': ')[1]) for val in s.split(' | ')[:4]])
    return res


def v_to_phs(vol: np.ndarray) -> np.ndarray:
    phs = -100 * vol
    return phs


class Device:
    def __init__(self, path_id: str):
        self.serial = serial.Serial(path_id)

    def spin(self, callback: Callable[[float, np.ndarray], None]):
        while self.serial.isOpen():
            s = self.serial.readline().decode("Ascii")
            if s.startswith('Voltage'):
                try:
                    arr = parse(s)
                except (ValueError, IndexError):
                    print("Error parsing data")
                    continue
                t = time.time()
                print(f"{t}: {arr}")
                callback(t, arr)
