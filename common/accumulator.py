import numpy as np


class Accumulator:

    def __init__(self):
        self.data = []

    def add(self, item):
        self.data.append(item)

    def avg(self):
        return np.mean(self.data)

    def avg_and_confidence_interval(self):
        # a = 1.0 * np.array(data)
        return self.avg(), np.std(self.data) * 1.96 / np.sqrt(len(self.data))

    def reset(self):
        self.data = []
