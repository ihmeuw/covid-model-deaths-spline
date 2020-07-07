import numpy as np

KNOT_DAYS_RATIO = 12
KNOT_DAYS_SYNTH = 16
FLOOR_DEATHS = 0.005


def get_data_se(data: np.array, log: bool = True) -> np.array:
    se = 1. / np.exp(data) ** 0.2
    
    return se
    

## ???
# def get_pseudo_nrmse(data: pd.DataFrame, 
#                      obs_var: str, pred_vars: List[str], 
#                      interval: int = 7) -> float:
#     # only days with both
#     no_na = data[[obs_var, pred_var]].notnull().all(axis=1)
#     data = data.loc[no_na, [obs_var, pred_var]]
    
#     # convert to daily
#     data = np.diff(data, axis=0, prepend=0)
    
#     # keep specified days
#     data = data[-interval:,:]
    
#     # get mean, then get rmse
#     data_range = data[:,0].ptp()
#     rmse = np.sqrt(np.mean(np.diff(data, axis=1)**2))
#     nrmse = rmse / data_range
    
#     return nrmse
