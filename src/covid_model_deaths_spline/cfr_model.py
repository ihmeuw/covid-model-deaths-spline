from pathlib import Path
from typing import Callable, List
import sys

from covid_shared import shell_tools
import dill as pickle
import numpy as np
import pandas as pd
import tqdm
import yaml

from covid_model_deaths_spline.mr_spline import SplineFit


def cfr_model(df: pd.DataFrame,
              dep_var: str, spline_var: str, indep_vars: List[str],
              model_dir: str,
              model_type: str,
              dow_holdout: int,
              daily: bool = False, log: bool = False) -> pd.DataFrame:
    # set up model
    df = df.copy()

    # add intercept
    df['intercept'] = 1

    # log transform, setting floor of 0.05 per population
    df = df.sort_values('Date').reset_index(drop=True)
    floor = 0.05 / df['population'].values[0]
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
    
    # check assumptions
    if daily:
        raise ValueError('Not expecting daily CFR/HFR model.')

    # keep what we can use to predict (subset further to fitting dataset below)
    non_na = ~df[list(adj_vars.values())[1:]].isnull().any(axis=1)
    df = df.loc[non_na].reset_index(drop=True)

    # lose NAs in deaths as well for modeling
    mod_df = df.copy()
    non_na = ~mod_df[adj_vars[dep_var]].isnull()
    mod_df = mod_df.loc[non_na,
                        ['intercept'] + list(adj_vars.values())].reset_index(drop=True)
    
    # only run if at least a week of observations
    if len(mod_df) >= 7:
        # determine knots
        n_model_days = len(mod_df)
        n_i_knots = max(int(n_model_days / 8) - 1, 3)
        
        # spline settings
        spline_options = {
            'spline_knots_type': 'domain',
            'spline_degree': 3,
            'spline_r_linear':True,
            'spline_l_linear':True,
            'prior_beta_uniform': np.array([[0, np.inf]] * (n_i_knots + 1)).T
        }
        if not daily:
            spline_options.update({'prior_spline_monotonicity':'increasing'})
        
        # conditional settings
        if log:
            add_args = {'scale_se_floor_pctile': 0.}
        else:
            add_args = {'se_default': np.sqrt(mod_df[adj_vars[dep_var]].max())}

        # run model
        mr_model = SplineFit(
            data=mod_df,
            dep_var=adj_vars[dep_var],
            spline_var=adj_vars[spline_var],
            indep_vars=['intercept'] + list(map(adj_vars.get, indep_vars)),
            n_i_knots=n_i_knots,
            spline_options=spline_options,
            scale_se=log,
            log=log,
            **add_args
        )
        mr_model.fit_model()
        prediction = mr_model.predict(df)
    else:
        prediction = np.array([np.nan] * len(df))
    
    # attach prediction
    df['Predicted model death rate'] = prediction
    df[f'Predicted death rate ({model_type})'] = df['Predicted model death rate']
    if log:
        df[f'Predicted death rate ({model_type})'] = np.exp(df[f'Predicted death rate ({model_type})'])
    if daily:
        df[f'Predicted death rate ({model_type})'] = df[f'Predicted death rate ({model_type})'].cumsum()

    with open(f"{model_dir}/{df['location_id'][0]}_{model_type}_{dow_holdout}.pkl", 'wb') as fwrite:
        pickle.dump(mr_model, fwrite, -1)

    return df
