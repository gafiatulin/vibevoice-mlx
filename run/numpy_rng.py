"""NumPy-compatible RNG for reproducible diffusion seeds.

Wraps np.random.RandomState to match the original pipeline's seed generation.
Each diffusion step uses rng.randint(2**31) as the inner seed.
"""

import numpy as np


class NumpyRNG:
    """NumPy RandomState wrapper for reproducible seed sequences."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def randint(self, high: int = 2**31) -> int:
        return int(self.rng.randint(0, high))
