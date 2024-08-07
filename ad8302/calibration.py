import numpy as np

from ad8302.arrays import create_grably_array
from ad8302.beamformer import Beamformer
from ad8302.device import Device, v_to_phs, ParseResult, v_to_mag
from ad8302.spacial import calc_phase_diff_np

NUM_ANTS = 4
NUM_SAMPLES = 50
idx = 0


def collect_calib_data():
    dev = Device('/dev/ttyACM0')
    calib_data = np.empty([NUM_SAMPLES, NUM_ANTS])

    def store_sample(_: float, parse: ParseResult):
        global idx
        phases = v_to_phs(parse.vphs)
        calib_data[idx] = phases
        idx += 1
        print(f"Collected {idx}/{NUM_SAMPLES}")

    global idx
    while idx != NUM_SAMPLES:
        dev.spin_once(store_sample)
    return calib_data


def main():
    src_loc = np.array([[0.0, 5]])

    print(f"Place source at {src_loc} meter from radat")
    freq = 0.433e9
    ants_loc, topology = create_grably_array(freq)
    calib_data = collect_calib_data()
    calib_data_mean = calib_data.mean(axis=0)
    expected_diff = calc_phase_diff_np(src_loc, ants_loc, topology, freq)
    print(expected_diff - calib_data_mean)


if __name__ == '__main__':
    main()
