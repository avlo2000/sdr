import numpy as np

from ad8302.beamformer import Beamformer
from ad8302.device import Device, v_to_phs, ParseResult, v_to_mag
from ad8302.spacial import calc_phase_diff

NUM_ANTS = 4
NUM_SAMPLES = 1000
idx = 0


def collect_calib_data():
    dev = Device('/dev/ttyACM0')
    calib_data = np.empty([NUM_SAMPLES, NUM_ANTS])

    def store_sample(_: float, parse: ParseResult):
        global idx
        phases = v_to_phs(parse.vphs)
        calib_data[idx] = phases
        idx += 1

    global idx
    while idx != NUM_SAMPLES:
        dev.spin_once(store_sample)
    return calib_data


def main():
    src_loc = np.array([[130.0, 100.0]])
    print("Place source at [1, 1] meter from radat")
    freq = 0.433e9

    ref_loc = np.array([[0.0, 0.0]])
    src_loc = np.array([[130.0, 100.0]])
    ants_loc = np.array(
        [[0.2, 0.0],
         [0.1, 0.0],
         [-0.1, 0.0],
         [-0.2, 0.0]]
    )
    calib_data = collect_calib_data()
    calib_data_mean = calib_data.mean(axis=0)
    expected_diff = calc_phase_diff(src_loc, ants_loc, ref_loc, freq)
    print(calib_data_mean - expected_diff)


if __name__ == '__main__':
    main()
