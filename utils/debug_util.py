import numpy as np


class ReservoirSampler(object):
    """Finds a random subset k elements from a stream of data in O(k) space.
    See https://en.wikipedia.org/wiki/Reservoir_sampling.
    """

    def __init__(self, k):
        self.samples = []
        self.num_seen = 0
        self.k = k

    def add(self, item):
        self.num_seen += 1

        if self.num_seen <= self.k:
            self.samples.append(item)
        elif np.random.rand(1)[0] <= self.k / (1.0 * self.num_seen):
            self.samples[np.random.choice(range(self.k))] = item

    def get_sample(self):
        return self.samples[:]
