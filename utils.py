import numpy as np


def shuffle(samples):
    x = np.random.permutation(samples)
    for i in range(5):
        x = x[np.random.permutation(samples)]
    return x
