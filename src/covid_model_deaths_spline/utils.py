import numpy as np

KNOT_DAYS = 16
FLOOR_DEATHS = 0.005


def get_ln_data_se(data: np.array) -> np.array:
    se = 1. / np.exp(data) ** 0.2
    
    return se
    