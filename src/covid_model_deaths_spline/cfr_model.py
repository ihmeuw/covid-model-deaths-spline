from typing import List

import dill as pickle
from loguru import logger
import numpy as np
import pandas as pd

from covid_model_deaths_spline.mr_spline import SplineFit
from covid_model_deaths_spline.utils import KNOT_DAYS, FLOOR_DEATHS, get_data_se


def run_model(mod_df: pd.DataFrame, df: pd.DataFrame, spline_knots_type: str,
              adj_vars: List[str], dep_var: str, spline_var: str, indep_vars: List[str],
              daily: bool, log: bool, model_type: str):
    # only run if at least a week of observations
    prediction = np.array([np.nan] * len(df))
    mr_model = None

    # determine knots
    n_model_days = len(mod_df)
    n_i_knots = max(int(n_model_days / KNOT_DAYS) - 1, 3)

    # spline settings
    spline_options = {
        'spline_knots_type': spline_knots_type,
        'spline_degree': 3,
        'spline_r_linear':True,
        'spline_l_linear':False
    }
    if not daily:
        spline_options.update({'prior_spline_monotonicity':'increasing'})

    # data SE
    if log:
        mod_df['obs_se'] = get_data_se(mod_df[adj_vars[dep_var]].values)
    else:
        mod_df['obs_se'] = np.sqrt(mod_df[adj_vars[dep_var]].max())

    # run model (if failure, might be because too many knots and constant case/hosp values; try again with fewer)
    prediction_pending = True
    while prediction_pending:
        try:
            mr_model = SplineFit(
                data=mod_df,
                dep_var=adj_vars[dep_var],
                spline_var=adj_vars[spline_var],
                indep_vars=['intercept'] + list(map(adj_vars.get, indep_vars)),
                n_i_knots=n_i_knots,
                spline_options=spline_options,
                log=log
            )
            mr_model.fit_model()
            prediction = mr_model.predict(df)
            if not np.isnan(prediction).any():
                prediction_pending = False
            else:
                raise ValueError('Prediction all nans (non-convergence).')
        except Exception as e:
            logger.warning(f'{model_type} model failed with {n_i_knots} knots.')
            logger.warning(f'Error: {e}')
        if n_i_knots == 2:
            prediction_pending = False
        if prediction_pending:
            n_i_knots -= 1

    # get rmse
    if mr_model is None:
        rmse = np.inf
    else:
        in_sample_prediction = mr_model.predict(mod_df)
        rmse = np.sqrt(np.mean((in_sample_prediction - mod_df[adj_vars[dep_var]])**2))

    return prediction, mr_model, rmse


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

    # log transform, setting floor of 0.005 per population
    df = df.sort_values('Date').reset_index(drop=True)
    floor = FLOOR_DEATHS / df['population'].values[0]
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
    # if not log:
    #     raise ValueError('Expecting log CFR/HFR model.')
    if daily:
        raise ValueError('Not expecting daily CFR/HFR model.')

    # keep what we can use to predict (subset further to fitting dataset below)
    non_na = ~df[list(adj_vars.values())[1:]].isnull().any(axis=1)
    df = df.loc[non_na].reset_index(drop=True)

    # only keep from last day of 2 cases ("imported case" threshold; prevents some duplicate values in spline)
    last_day_of_two_cases = (df[spline_var][::-1] <= 2 / df['population'][0]).cumsum()[::-1] <= 1
    last_day_of_zero_deaths = (df[spline_var][::-1] == 0 / df['population'][0]).cumsum()[::-1] <= 1
    df = df.loc[last_day_of_two_cases & last_day_of_zero_deaths].reset_index(drop=True)

    # don't predict deaths before deaths data
    has_deaths = df[dep_var].notnull()
    has_deaths = np.cumsum(has_deaths)
    has_deaths = has_deaths > 0
    df = df.loc[has_deaths].reset_index(drop=True)

    # lose all NAs in deaths for modeling
    mod_df = df.copy()
    non_na = ~mod_df[adj_vars[dep_var]].isnull()
    mod_df = mod_df.loc[non_na,
                        ['intercept'] + list(adj_vars.values())].reset_index(drop=True)

    # model with domain- and frequency-based knots, use the better fit of the two
    domain_prediction, domain_model, domain_rmse = run_model(mod_df.copy(), df.copy(), 'domain',
                                                adj_vars, dep_var, spline_var, indep_vars,
                                                daily, log, model_type)
    frequency_prediction, frequency_model, freq_rmse = run_model(mod_df.copy(), df.copy(), 'frequency',
                                                                 adj_vars, dep_var, spline_var, indep_vars,
                                                                 daily, log, model_type)
    if domain_rmse < freq_rmse:
        prediction, mr_model = domain_prediction, domain_model
    else:
        prediction, mr_model = frequency_prediction, frequency_model

    # attach prediction
    df['Predicted model death rate'] = prediction
    df[f'Predicted death rate ({model_type})'] = df['Predicted model death rate']
    if log:
        df[f'Predicted death rate ({model_type})'] = np.exp(df[f'Predicted death rate ({model_type})'])
    if daily:
        df[f'Predicted death rate ({model_type})'] = df[f'Predicted death rate ({model_type})'].cumsum()

    # save
    if mr_model is not None:
        with open(f"{model_dir}/{df['location_id'][0]}_{model_type}_{dow_holdout}.pkl", 'wb') as fwrite:
            pickle.dump(mr_model.mr_model, fwrite, -1)

    return df
