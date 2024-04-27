import adi
import numpy as np


def create_chirp(sdr):
    pulse_duration = 140e-6  # sec
    num_samps = int(sdr.sample_rate * pulse_duration)
    if ~(num_samps % 1024):
        num_samps = 1024 * (num_samps // 1024)
    if num_samps == 0:
        num_samps = 1024
    print(num_samps)
    bandwidth = sdr.tx_rf_bandwidth  # 20e6

    time_axis = np.linspace(0, pulse_duration, num_samps)
    f0 = - bandwidth / 2

    alpha = bandwidth / pulse_duration

    trans_chirp = np.exp(1j * 2 * np.pi * (f0 * time_axis + (alpha * time_axis ** 2) / 2))
    trans_chirp *= 2 * 14
    return trans_chirp


def create_rnd_signal(num_symbols: int):
    """ Creates transmit waveform (QPSK, 16 samples per symbol)
    :param num_symbols: number of symbols in signal
    :return:
    """
    x_int = np.random.randint(0, 4, num_symbols)  # 0 to 3
    x_degrees = x_int * 360 / 4.0 + 45  # 45, 135, 225, 315 degrees
    x_radians = x_degrees * np.pi / 180.0  # sin() and cos() takes in radians
    x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians)  # this produces our QPSK complex symbols
    samples = np.repeat(x_symbols, 16)  # 16 samples per symbol (rectangular pulses)
    samples *= 2 ** 15  # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    return samples


def take_sample(sdr: adi.Pluto, freq: int, signal: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    :param sdr: Pluto sdr device to use
    :param freq: center carrier frequency for the sample
    :param signal: signal to transmit
    :return: signal received
    """
    assert sdr.tx_cyclic_buffer

    # sdr.rx_lo = freq
    # sdr.tx_lo = freq

    sdr.tx(signal)
    rx = sdr.rx()
    sdr.tx_destroy_buffer()
    return rx
