import numpy as np

KNOT_DAYS = 16
FLOOR_DEATHS = 0.005


def get_data_se(data: np.array, log: bool = True) -> np.array:
    if log:
        se = 1. / np.exp(data) ** 0.2
    else:
        raise ValueError('No linear trasformation.')
    
    return se
