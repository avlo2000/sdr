import dataclasses
import time
from typing import Callable

import numpy as np
import serial
import re


@dataclasses.dataclass(init=False)
class ParseResult:
    vphs: np.ndarray
    vmag: np.ndarray


def parse(s: str) -> ParseResult:
    res = ParseResult()
    vmag_matches = re.findall(r'VMAG[0-3]:\s*(-?\d*\.\d+|-?\d+\.?)', s)
    vphs_matches = re.findall(r'VPHS[0-3]:\s*(-?\d*\.\d+|-?\d+\.?)', s)
    res.vmag = np.array([float(val) for val in vmag_matches])
    res.vphs = np.array([float(val) for val in vphs_matches])
    if not len(res.vphs) == len(res.vmag) == 4:
        raise RuntimeError("Wrong len")
    return res


def v_to_phs(vol: np.ndarray) -> np.ndarray:
    phs = -100 * (vol - 0.9) + 90
    return phs


def v_to_mag(vol: np.ndarray) -> np.ndarray:
    mag = (vol - 0.9) / 0.03
    return mag


class Device:
    def __init__(self, path_id: str):
        self.serial = serial.Serial(path_id)

    def spin(self, callback: Callable[[float, ParseResult], None]):
        while True:
            s = self.serial.readline().decode("Ascii")
            try:
                arr = parse(s)
            except (ValueError, IndexError, RuntimeError):
                print("Error parsing data")
                continue
            t = time.time()
            print(f"{t}: {arr}")
            callback(t, arr)
