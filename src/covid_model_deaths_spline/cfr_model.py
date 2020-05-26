import numpy as np
import pandas as pd
import dill as pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from smoother import smoother
from mr_spline import SplineFit


def cfr_model(df: pd.DataFrame, deaths_threshold: int, 
              daily: bool, log: bool, 
              dep_var: str, spline_var: str, indep_vars: List[str],
              model_dir: str) -> pd.DataFrame:
    # add intercept
    df['intercept'] = 1

    # log transform, setting floor of 0.01 per population
    df = df.sort_values('Date').reset_index(drop=True)
    floor = 0.01 / df['population'].values[0]
    adj_vars = {}
    for orig_var in [dep_var, spline_var] + indep_vars:
        mod_var = f'Model {orig_var.lower()}'
        df[mod_var] = df[orig_var]
        if daily:
            start_idx = df.loc[~df[mod_var].isnull()].index.values[0]
            df[mod_var][start_idx+1:] = np.diff(df[mod_var].values[start_idx:])
        if log:
            df.loc[df[mod_var] < floor, mod_var] = floor
            df[mod_var] = np.log(df[mod_var])
        adj_vars.update({orig_var:mod_var})
    df['Model log'] = log
    df['Model daily'] = daily

    # keep what we can use to predict (subset further to fitting dataset below)
    non_na = ~df[list(adj_vars.values())[1:]].isnull().any(axis=1)
    df = df.loc[non_na].reset_index(drop=True)

    # lose NAs in deaths as well for modeling
    mod_df = df.copy()
    above_thresh = (mod_df[dep_var] * df['population']) >= deaths_threshold
    has_x = (mod_df[spline_var] * df['population']) >= 1
    non_na = ~mod_df[adj_vars[dep_var]].isnull()
    mod_df = mod_df.loc[above_thresh & has_x & non_na, ['intercept'] + list(adj_vars.values())].reset_index(drop=True)
    if len(mod_df) < 3:
        raise ValueError(f"Fewer than 3 days {deaths_threshold}+ deaths and 1+ cases in {df['location_name'][0]}")

    # run model and predict
    if (df[dep_var] * df['population']).max() > 20:
        # cubic spline with linear tails
        n_i_knots = 2
        spline_options={
                'spline_knots_type': 'frequency',
                'spline_degree': 3,
                'spline_r_linear':True,
                'spline_l_linear':True,
                'prior_beta_uniform':np.array([0., np.inf]),
            }
    else:
        # linear spline
        n_i_knots = 1
        spline_options={
                'spline_knots_type': 'frequency',
                'spline_degree': 1,
                'prior_beta_uniform':np.array([0., np.inf]),
            }

    if not daily:
        spline_options.update({'prior_spline_monotonicity':'increasing'})
    mr_mod = SplineFit(
        data=mod_df, 
        dep_var=adj_vars[dep_var],
        spline_var=adj_vars[spline_var],
        indep_vars=['intercept'] + list(map(adj_vars.get, indep_vars)),
        n_i_knots=n_i_knots,
        spline_options=spline_options,
        scale_se=False
    )
    mr_mod.fit_model()
    df['Predicted model death rate'] = mr_mod.predict(df)
    df['Predicted death rate'] = df['Predicted model death rate']
    if log:
        df['Predicted death rate'] = np.exp(df['Predicted death rate'])
    if daily:
        df['Predicted death rate'] = df['Predicted death rate'].cumsum()
    
    with open(f"{model_dir}/{df['location_id'][0]}.pkl", 'wb') as fwrite:
        pickle.dump(mr_mod, fwrite, -1)
    
    return df
