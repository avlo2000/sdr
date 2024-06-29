import json

import numpy as np

from ad8302.device import Device, v_to_phs, ParseResult

NUM_ANTS = 4
NUM_SAMPLES = 1000
idx = 0


def collect_phs_data() -> np.ndarray:
    data = np.empty([NUM_SAMPLES, NUM_ANTS])
    dev = Device('/dev/ttyACM0')

    def store_sample(_: float, parse: ParseResult):
        global idx
        phases = v_to_phs(parse.vphs)
        data[idx] = phases
        idx += 1
        print(f"Collected {idx}/{NUM_SAMPLES}")

    global idx
    while idx != NUM_SAMPLES:
        dev.spin_once(store_sample)
    idx = 0
    return data


def main():

    print("Input q/Q to exit and save data")
    data = []
    while True:
        inp = input("Input source location[Theta, Radius]: ")
        if inp in ('q', 'Q'):
            with open("data/dataset.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            return
        sample = dict()
        src_loc = np.array([float(x) for x in inp.split(' ')])
        phs_data = collect_phs_data()
        sample['src_loc'] = list(src_loc)
        for i in range(NUM_ANTS):
            sample[f'phs_data_{i}'] = list(phs_data[:, i])
        data.append(sample)


if __name__ == '__main__':
    main()
