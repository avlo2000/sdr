import numpy as np

from ad8302.beamformer import Beamformer
from ad8302.device import Device, v_to_phs, ParseResult, v_to_mag

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
    d_to_ref = np.array([0.205, 0.104, 0.100, 0.202])

    calib_data = collect_calib_data()
    phase_mean = calib_data.mean(axis=0)



if __name__ == '__main__':
    main()
