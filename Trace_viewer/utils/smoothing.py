import numpy as np
from scipy.ndimage import uniform_filter1d

def uf(t, lag, axis=-1):
    """
    Apply a uniform filter (moving average) on array t.
    """
    return uniform_filter1d(t, size=lag, mode='nearest', axis=axis)

def sa(t, lag, axis = -1):
    """
    Apply strided averaging on array t.
    """
    t = t[:t.shape[0] - (t.shape[0] % lag)]
    t = t.reshape(-1, lag)
    return np.average(t, axis=1)
