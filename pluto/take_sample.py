import adi
import numpy as np


def take_sample(sdr: adi.Pluto, freq: int, signal: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    :param sdr: Pluto sdr device to use
    :param freq: center frequency for the sample
        :param signal:
    :return: reference transmitted signal, signal received
    """
    assert sdr.tx_cyclic_buffer

    sdr.rx_lo = freq
    sdr.tx_lo = freq

    sdr.tx(signal)
    rx = None
    for _ in range(1):
        rx = sdr.rx()
    sdr.tx_destroy_buffer()
    return signal, rx
