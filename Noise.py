import numpy as np

class Noise:
    def __init__(self):
        pass

    def __call__(self):
        return np.random.randn() * 0.01