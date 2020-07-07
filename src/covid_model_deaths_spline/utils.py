import numpy as np

KNOT_DAYS_RATIO = 12
KNOT_DAYS_SYNTH = 16
FLOOR_DEATHS = 0.005


def get_data_se(data: np.array, log: bool = True) -> np.array:
    se = 1. / np.exp(data) ** 0.2
    
    return se
    