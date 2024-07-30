import json
from collections import deque
from time import sleep

import numpy as np
from matplotlib import pyplot as plt

from ad8302.device import Device, v_to_phs, ParseResult

NUM_ANTS = 4
NUM_SAMPLES = 200
idx = 0


ax = plt.subplot(111)
fig = plt.figure()
plt.grid(True)
plt.legend()

plt.show(block=False)
phs_lines = []
n = 200
time_data = deque(maxlen=NUM_SAMPLES)
phase_data = deque(maxlen=NUM_SAMPLES)
ax.set_xlim([0, phase_data.maxlen])
ax.set_ylim([-190.0, +190.0])
for i in range(NUM_ANTS):
    line, = ax.plot([], [])
    phs_lines.append(line)


def collect_phs_data() -> np.ndarray:
    data = np.empty([NUM_SAMPLES, NUM_ANTS])
    dev = Device('/dev/ttyACM0')

    def store_sample(_: float, parse: ParseResult):
        global idx
        phases = v_to_phs(parse.vphs)
        data[idx] = phases
        idx += 1
        phase_data.append(phases)

        for j, y in enumerate(np.array(phase_data).T):
            phs_lines[j].set_data(time_data, y)
        print(f"Collected {idx}/{NUM_SAMPLES}")

        fig.canvas.draw()
        fig.canvas.flush_events()
        sleep(0.01)

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
