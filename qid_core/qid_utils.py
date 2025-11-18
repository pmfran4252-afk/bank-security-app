import numpy as np


def safe_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.clip(x, a_min=0.0, a_max=None))
