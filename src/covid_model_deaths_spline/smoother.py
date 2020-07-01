import functools
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple
from collections import namedtuple
import dill as pickle

import numpy as np
import pandas as pd
import tqdm

from covid_model_deaths_spline.mr_spline import SplineFit, rescale_k


def apply_floor(vals: np.array, floor_val: float) -> np.array:
    vals[vals < floor_val] = floor_val

    return vals


def run_smoothing_model(mod_df: pd.DataFrame, n_i_knots: int, spline_options: Dict, scale_se: bool,
                        pred_df: pd.DataFrame, ensemble_knots: np.array = None, 
                        results_only: bool = False, se_default: float = 1.,
                        log: bool = True) -> np.array:
    mr_model = SplineFit(
        data=mod_df,
        dep_var='y',
        spline_var='x',
        indep_vars=['intercept'],
        n_i_knots=n_i_knots,
        spline_options=spline_options,
        ensemble_knots=ensemble_knots,
        scale_se=scale_se,
        observed_var='observed',
        pseudo_se_multiplier=2.,
        se_default=se_default,
        log=log
    )
    mr_model.fit_model()
    smooth_y = mr_model.predict(pred_df)
    mod_df = mr_model.data
    mod_df['smooth_y'] = mr_model.predict(mod_df)

    if results_only:
        return smooth_y
    else:
        return smooth_y, mr_model.mr_model, mod_df


def process_inputs(y: np.array, col_names: List[str], 
                   x: np.array, n_i_knots: int, subset_idx: np.array,
                   mono: bool, limits: np.array, tail_gprior: np.array = None):
    # get smoothed curve (dropping NAs, inflating variance for pseudo-deaths)
    obs_data = y.copy()
    obs_data[:,0] = 1
    obs_data[:,1:] = 0
    obs_data[~subset_idx] = 0
    y_fit = y.flatten()
    obs_data = obs_data.flatten()
    col_data = np.array([col_names] * y.shape[0]).flatten()
    x_fit = np.repeat(x, y.shape[1], axis=0)
    non_na_idx = ~np.isnan(y_fit)
    y_fit = y_fit[non_na_idx]
    obs_data = obs_data[non_na_idx]
    col_data = col_data[non_na_idx]
    x_fit = x_fit[non_na_idx]
    mod_df = pd.DataFrame({
        'y':y_fit,
        'intercept':1,
        'x':x_fit,
        'data_type':col_data,
        'observed':obs_data
    })
    mod_df['observed'] = mod_df['observed'].astype(bool)
    beta_prior = np.array([limits] * (n_i_knots + 1))
    beta_prior[0] = np.array([0, np.inf]) # no intercept, so actually first beta
    spline_options={
            'spline_knots_type': 'domain',
            'spline_degree': 3,
            'spline_r_linear': True,
            'spline_l_linear': True,
            'prior_beta_uniform': beta_prior.T
        }
    if mono:
        spline_options.update({'prior_spline_monotonicity': 'increasing'})
    if tail_gprior is not None:
        maxder_gprior = np.array([[0, np.inf]] * (n_i_knots + 1)).T
        if tail_gprior.size != 2:
            raise ValueError('`tail_gprior` must be in the format np.array([mu, sigma])')
        maxder_gprior[:,-1] = tail_gprior
        spline_options.update({'prior_spline_maxder_gaussian': maxder_gprior})

    return mod_df, spline_options


def draw_cleanup(draws: np.array, smooth_y: np.array, x: np.array, df: pd.DataFrame) -> pd.DataFrame:
    # set to linear, add up cumulative, and create dataframe
    draws -= np.var(draws, axis=1, keepdims=True) / 2
    draws = np.exp(draws)
    #draws *= np.exp(smooth_y) / draws.mean(axis=1, keepdims=True)
    draws[draws * df['population'].values[0] < 1e-10] = 1e-10 / df['population'].values[0]
    draws = draws.cumsum(axis=0)

    # store in dataframe
    draw_df = pd.DataFrame({
        'location_id': df['location_id'].unique().item(),
        'Date': [df['Date'].min() + pd.Timedelta(days=dx) for dx in x],
        'population': df['population'].unique().item()
    })
    draw_df = pd.concat([draw_df,
                         pd.DataFrame(draws, columns=[f'draw_{d}' for d in range(draws.shape[1])])], axis=1)

    return draw_df


def get_limits(y: np.array) -> np.array:
    y = y.copy()
    y = y[~np.isnan(y)]
    upper_lim = np.abs(y).max()
    lower_lim = -upper_lim
    
    return np.array([lower_lim, upper_lim])


def combine_cumul_daily(from_cumul: np.array, from_daily: np.array, 
                        total_deaths: float, ceiling: float = 50.) -> np.array:
    # convert cumulative to daily (replace first day with first estimate)
    from_cumul = np.exp(from_cumul.copy())
    from_cumul[1:] = np.diff(from_cumul, axis=0)
    from_cumul = apply_floor(from_cumul, min(from_cumul[from_cumul > 0]))
    
    # daily in linear
    from_daily = np.exp(from_daily.copy())
    
    # weighted combination
    d_w = 0.  # min(total_deaths / ceiling, 1.)
    c_w = 1. - d_w
    smooth_y = from_daily * d_w + from_cumul * c_w
    
    return np.log(smooth_y)


def get_mad(df: pd.DataFrame, weighted: bool) -> float:
    residuals = (df['y'] - df['smooth_y_insample']).values
    abs_residuals = np.abs(residuals)
    if weighted:
        weights = (1 / df['obs_se']**2).values
        weights /= weights.sum()
        weights = weights[np.argsort(abs_residuals)]
        abs_residuals = np.sort(abs_residuals)
        w_cumul = weights.cumsum()
        mad = abs_residuals[w_cumul >= 0.5][0]
    else:
        mad = np.median(abs_residuals)
    
    return mad


def find_best_settings(mr_model, spline_options):
    x = mr_model.data.df['x'].values
    knots = x.min() + mr_model.ensemble_knots * x.ptp()
    betas = [mr.beta_soln for mr in mr_model.sub_models]
    best_knots = np.average(knots, axis=0, weights=mr_model.weights)
    best_betas = np.average(betas, axis=0, weights=mr_model.weights)
    best_betas[1:] += best_betas[0]
    
    Results = namedtuple('Results', 'knots betas options')
    
    return Results(best_knots, best_betas, spline_options)


def smoother(df: pd.DataFrame, obs_var: str, pred_vars: List[str],
             n_draws: int, dow_holdout: int,
             floor_deaths: float = 0.001) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # extract inputs (only keep one day of 0)
    df = df.sort_values('Date').reset_index(drop=True)
    total_deaths = (df['Death rate'] * df['population']).max()
    floor = floor_deaths / df['population'][0]
    keep_idx = ~df[[obs_var] + pred_vars].isnull().all(axis=1)
    max_1week_of_zeros_head = (df[obs_var][::-1] == 0).cumsum()[::-1] <= 7
    cumul_y = df.loc[keep_idx, [obs_var] + pred_vars].values
    daily_y = cumul_y.copy()
    daily_y[1:] = np.diff(daily_y, axis=0)
    #ln_cumul_y = np.log(apply_floor(cumul_y, floor))
    ln_daily_y = np.log(apply_floor(daily_y, floor))
    x = df.index[keep_idx].values
    
    # number of knots
    n_model_days = len(df.loc[df[obs_var].notnull() & max_1week_of_zeros_head])
    n_i_knots = max(int(n_model_days / 8) - 1, 3)

    # get deaths in last week to determine flat prior for daily
    interval_n = 14
    interval_denom = interval_n * 10.
    last_interval = df.copy()
    last_interval['Deaths'] = last_interval['Death rate'] * last_interval['population']
    last_interval['Deaths'][1:] = np.diff(last_interval['Deaths'])
    last_interval_deaths = last_interval.loc[~last_interval['Deaths'].isnull()].iloc[-interval_n:]['Deaths'].sum()
    daily_gprior_std = max(floor_deaths, last_interval_deaths) / interval_denom

    # prediction
    pred_df = pd.DataFrame({'intercept':1, 'x': x})
    
    # prepare daily data
    ln_daily_limits = get_limits(ln_daily_y[max_1week_of_zeros_head])
    ln_daily_mod_df, ln_daily_spline_options = process_inputs(
        y=ln_daily_y, col_names=[obs_var] + pred_vars,
        x=x, n_i_knots=n_i_knots, subset_idx=max_1week_of_zeros_head,
        mono=False, limits=ln_daily_limits, tail_gprior=np.array([0, daily_gprior_std])
    )
    if ln_daily_mod_df['x'].unique().size < n_i_knots * 3:
        raise ValueError(f'Smoother model data (daily) contains fewer than {n_i_knots * 3} days.')
    ln_daily_smooth_y, ln_daily_model, ln_daily_mod_df = run_smoothing_model(
        ln_daily_mod_df, n_i_knots, ln_daily_spline_options, True, pred_df
    )
    # ensemble_knots = ln_daily_model.ensemble_knots
    
    # prepare cumulative data and run model
    cumul_limits = np.array([0., np.inf]) # get_limits(ln_cumul_y[max_1week_of_zeros_head])
    cumul_mod_df, cumul_spline_options = process_inputs(
        y=cumul_y, col_names=[obs_var] + pred_vars,
        x=x, n_i_knots=n_i_knots, subset_idx=max_1week_of_zeros_head,
        mono=True, limits=cumul_limits, tail_gprior=np.array([0, np.inf])
    )
    if cumul_mod_df['x'].unique().size < n_i_knots * 5:
        raise ValueError(f'Smoother model data (cumulative) contains fewer than {n_i_knots * 5} days.')
    cumul_smooth_y, cumul_model, cumul_mod_df = run_smoothing_model(
        cumul_mod_df, n_i_knots, cumul_spline_options, False, pred_df,  # , ensemble_knots
        se_default=np.sqrt(cumul_mod_df['y'].max()),
        log=False
    )
    ensemble_knots = cumul_model.ensemble_knots
    
    # set floor and convert to log dailiy
    smooth_y = cumul_smooth_y.copy()
    smooth_y[smooth_y < floor] = floor
    smooth_y[1:] = np.diff(smooth_y, axis=0)
    smooth_y = apply_floor(smooth_y, floor)
    smooth_y = np.log(smooth_y)
    
    # same for dataset prediction
    smooth_y_insample = pd.pivot_table(cumul_mod_df, 
                                       index='x', columns='data_type', values='smooth_y').values
    smooth_y_insample[smooth_y_insample < floor] = floor
    smooth_y_insample[1:] = np.diff(smooth_y_insample, axis=0)
    smooth_y_insample = apply_floor(smooth_y_insample, floor)
    smooth_y_insample = np.log(smooth_y_insample)
    smooth_y_insample = smooth_y_insample.flatten()
    ln_daily_mod_df['smooth_y_insample'] = smooth_y_insample[~np.isnan(smooth_y_insample)]

    # get uncertainty in ln(daily)
    mad = get_mad(ln_daily_mod_df, weighted=True)
    rstd = mad * 1.4826
    smooth_y = np.array([smooth_y]).T
    noisy_draws = np.random.normal(smooth_y, rstd, (smooth_y.size, n_draws))

    # refit in ln(daily)
    draw_mod_dfs = [
        pd.DataFrame({
            'y':nd,
            'intercept':1,
            'x':x,
            'observed':True
        })
        for nd in noisy_draws.T
    ]
    x_pred = x.max() + np.arange(dow_holdout + 1)[1:]
    x_pred = np.hstack([x, x_pred])
    refit_pred_df = pd.DataFrame({'intercept':1, 'x': x_pred})
    refit_se = 1 / np.exp(smooth_y)**0.2
    refit_se = np.mean(refit_se[-15:])
    rescaled_ensemble_knots = rescale_k(ln_daily_mod_df['x'].values, x, ensemble_knots)
    _combiner = functools.partial(run_smoothing_model,
                                  n_i_knots=n_i_knots,
                                  spline_options=ln_daily_spline_options,
                                  pred_df=refit_pred_df,
                                  scale_se=False,
                                  se_default=refit_se,
                                  ensemble_knots=rescaled_ensemble_knots,
                                  results_only=True)
    with multiprocessing.Pool(20) as p:
        smooth_draws = list(tqdm.tqdm(p.imap(_combiner, draw_mod_dfs), total=n_draws))
    smooth_draws = np.vstack(smooth_draws).T

    # make pretty (in linear cumulative space)
    noisy_draws = draw_cleanup(noisy_draws, smooth_y, x_pred, df)
    smooth_draws = draw_cleanup(smooth_draws, smooth_y, x_pred, df)
    
    # get best knots and betas
    best_settings = find_best_settings(ln_daily_model, ln_daily_spline_options)

    return noisy_draws, smooth_draws, best_settings


def synthesize_time_series(df: pd.DataFrame,
                           obs_var: str, pred_vars: List[str],
                           spline_vars: List[str],
                           spline_settings_dir: str,
                           n_draws: int, dow_holdout: int) -> pd.DataFrame:
    # location data
    df = df.copy()

    # spline on deaths time series
    noisy_draws, smooth_draws, best_settings = smoother(
        df=df.copy(),
        obs_var=obs_var,
        pred_vars=pred_vars,
        n_draws=n_draws,
        dow_holdout=dow_holdout
    )
    
    # save knots
    with open(f"{spline_settings_dir}/{df['location_id'][0]}_{dow_holdout}.pkl", 'wb') as fwrite:
        pickle.dump(best_settings, fwrite, -1)

    return noisy_draws, smooth_draws
