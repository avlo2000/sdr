import numpy as np


class HistFilter:
    def __init__(self, hist_reduction: float):
        self._hist_reduction = hist_reduction

    def filter(self, data: np.ndarray):
        std = np.std(data)
        hist, bins = np.histogram(data, bins=int(len(data)/self._hist_reduction))
        most_probable = bins[np.argmax(hist)]
        filtered = data[abs((data - most_probable)) / std < 1.0]
        return filtered
