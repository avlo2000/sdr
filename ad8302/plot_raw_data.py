import time
from collections import deque

from matplotlib import pyplot as plt
import numpy as np

from ad8302.device import Device

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_ylim([0.0, 1.9])
plt.grid(True)

num_ant = 4
lines = []
for _ in range(num_ant):
    line, = ax.plot([], lw=3)
    lines.append(line)
fig.canvas.draw()
plt.show(block=False)
t_start = time.time()

x_data = deque(maxlen=2000)
y_data = deque(maxlen=2000)


def live_update(x_val: float, y_val: np.ndarray):
    x_data.append(x_val)
    y_data.append(y_val)
    ys = np.array(y_data)

    for i, y in enumerate(ys.T):
        lines[i].set_data(x_data, y)
    ax.set_xlim(max(x_data) - 10.0, max(x_data))
    fig.canvas.draw()
    fig.canvas.flush_events()


if __name__ == '__main__':
    dev = Device('COM5')
    dev.spin(live_update)

