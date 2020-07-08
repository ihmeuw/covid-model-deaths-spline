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
from covid_model_deaths_spline.utils import KNOT_DAYS_SYNTH, FLOOR_DEATHS, get_data_se
from covid_model_deaths_spline.cluster import F_THREAD


def apply_floor(vals: np.array, floor_val: float) -> np.array:
    vals[vals < floor_val] = floor_val

    return vals


def run_smoothing_model(mod_df: pd.DataFrame, n_i_knots: int, spline_options: Dict,
                        pred_df: pd.DataFrame, ensemble_knots: np.array, 
                        results_only: bool,
                        log: bool) -> np.array:
    mr_model = SplineFit(
        data=mod_df,
        dep_var='y',
        spline_var='x',
        indep_vars=['intercept'],
        n_i_knots=n_i_knots,
        spline_options=spline_options,
        ensemble_knots=ensemble_knots,
        observed_var='observed',
        pseudo_se_multiplier=2.,
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
                   x: np.array, n_i_knots: int,  #observed_days: np.array,
                   mono: bool, limits: np.array, tail_gprior: np.array):
    # get smoothed curve (dropping NAs, inflating variance for pseudo-deaths)
    obs_data = y.copy()
    obs_data[:,0] = 1
    obs_data[:,1:] = 0
    #obs_data[~observed_days] = 0
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

    # basic settings
    spline_options={
            'spline_knots_type': 'domain',
            'spline_degree': 3,
            'spline_r_linear': True,
            'spline_l_linear': True
        }
    
    # cumulative or daily
    if mono:
        # settings for cumulative
        spline_options.update({'prior_spline_monotonicity': 'increasing'})
    else:
        # settings for daily (no intercent, actually setting first beta increasing)
        beta_prior = np.array([limits] * (n_i_knots + 1))
        beta_prior[0] = np.array([0, np.inf])
        spline_options.update({'prior_beta_uniform': beta_prior.T})
        
        # penalize wiggliness
        maxder_gprior = np.array([[0, np.inf]] + [[0, 0.05]] * (n_i_knots - 1) + [[0, np.inf]]).T
        if tail_gprior.size != 2:
            raise ValueError('`tail_gprior` must be in the format np.array([mu, sigma])')
        maxder_gprior[:,-1] = tail_gprior
        spline_options.update({'prior_spline_maxder_gaussian': maxder_gprior})

    return mod_df, spline_options


def draw_cleanup(draws: np.array,
                 x: np.array, df: pd.DataFrame) -> pd.DataFrame:
    # set to linear, add up cumulative, and create dataframe
    draws -= np.var(draws, axis=1, keepdims=True) / 2
    draws = np.exp(draws)
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


def get_gprior_std(df: pd.DataFrame, last_interval_days: int = 7,
                   last_interval_deaths_threshold: int = 21) -> float:
    daily_df = df.copy()
    daily_df['Deaths'] = daily_df['Death rate'] * daily_df['population']
    daily_df['Deaths'][1:] = np.diff(daily_df['Deaths'])
    last_interval_deaths = daily_df.loc[~daily_df['Deaths'].isnull()].iloc[-last_interval_days:]['Deaths'].sum()
    if last_interval_deaths <= last_interval_deaths_threshold:
        gprior_std = 1e-4
    else:
        gprior_std = np.inf
        
    return gprior_std


def smoother(df: pd.DataFrame, obs_var: str, pred_vars: List[str],
             n_draws: int, dow_holdout: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # extract inputs
    df = df.sort_values('Date').reset_index(drop=True)
    total_deaths = (df['Death rate'] * df['population']).max()
    floor = FLOOR_DEATHS / df['population'][0]
    keep_idx = ~df[[obs_var] + pred_vars].isnull().all(axis=1)
    cumul_y = df.loc[keep_idx, [obs_var] + pred_vars].values
    ln_cumul_y = np.log(apply_floor(cumul_y.copy(), floor))
    daily_y = cumul_y.copy()
    daily_y_isna = np.isnan(daily_y)
    daily_y[daily_y_isna] = 0
    daily_y = np.diff(daily_y, axis=0, prepend=0)
    daily_y[daily_y_isna] = np.nan
    ln_daily_y = np.log(apply_floor(daily_y, floor))
    x = df.index[keep_idx].values
    
    # number of knots
    n_model_days = len(df.loc[df[obs_var].notnull()])
    n_i_knots = max(int(n_model_days / KNOT_DAYS_SYNTH) - 1, 3)

    # get deaths in last week to determine flat prior for daily
    gprior_std = get_gprior_std(df)

    # prediction
    pred_df = pd.DataFrame({'intercept':1, 'x': x})
    
    # prepare daily data
    ln_daily_limits = get_limits(ln_daily_y)
    ln_daily_mod_df, ln_daily_spline_options = process_inputs(
        y=ln_daily_y, col_names=[obs_var] + pred_vars,
        x=x, n_i_knots=n_i_knots,
        mono=False, limits=ln_daily_limits, tail_gprior=np.array([0, gprior_std])
    )
    ln_daily_mod_df['obs_se'] = get_data_se(ln_daily_mod_df['y'].values)
    
    # prepare cumulative data and run model
    cumul_mod_df, cumul_spline_options = process_inputs(
        y=cumul_y, col_names=[obs_var] + pred_vars,
        x=x, n_i_knots=n_i_knots,
        mono=True, limits=np.array([0., np.inf]), tail_gprior=np.array([0, gprior_std])
    )
    cumul_mod_df['obs_se'] = np.sqrt(cumul_mod_df['y'].max())
    cumul_smooth_y, cumul_model, cumul_mod_df = run_smoothing_model(
        cumul_mod_df, n_i_knots, cumul_spline_options, pred_df,
        ensemble_knots=None, results_only=False, log=False
    )
    ensemble_knots = cumul_model.ensemble_knots
    
    # set floor and convert to log daily
    ln_daily_smooth_y = cumul_smooth_y.copy()
    ln_daily_smooth_y[ln_daily_smooth_y < floor] = floor
    ln_daily_smooth_y = np.diff(ln_daily_smooth_y, axis=0, prepend=0)
    ln_daily_smooth_y = apply_floor(ln_daily_smooth_y, floor)
    ln_daily_smooth_y = np.log(ln_daily_smooth_y)
    
    # same for dataset prediction
    smooth_y_insample = pd.pivot_table(cumul_mod_df, 
                                       index='x', columns='data_type', values='smooth_y').values
    smooth_y_insample[smooth_y_insample < floor] = floor
    smooth_y_insample_isna = np.isnan(smooth_y_insample)
    smooth_y_insample[smooth_y_insample_isna] = 0
    smooth_y_insample = np.diff(smooth_y_insample, axis=0, prepend=0)
    smooth_y_insample[smooth_y_insample_isna] = np.nan
    smooth_y_insample = apply_floor(smooth_y_insample, floor)
    smooth_y_insample = np.log(smooth_y_insample)
    smooth_y_insample = smooth_y_insample.flatten()
    ln_daily_mod_df['smooth_y_insample'] = smooth_y_insample[~np.isnan(smooth_y_insample)]

    # sample residual noise in ln(daily), convert back to linear cumul
    mad = get_mad(ln_daily_mod_df, weighted=True)
    rstd = mad * 1.4826
    ln_daily_smooth_y = np.array([ln_daily_smooth_y]).T
    noisy_draws = np.random.normal(ln_daily_smooth_y, rstd, (ln_daily_smooth_y.size, n_draws))

    # refit data
    draw_mod_dfs = [
        pd.DataFrame({
            'y':nd,
            'intercept':1,
            'x':x,
            'observed':True,
            'obs_se':get_data_se(ln_daily_smooth_y.flatten())
        })
        for nd in noisy_draws.T
    ]
    x_pred = x.max() + np.arange(dow_holdout + 1)[1:]
    x_pred = np.hstack([x, x_pred])
    refit_pred_df = pd.DataFrame({'intercept':1, 'x': x_pred})
    
    # refit settings
    rescaled_ensemble_knots = rescale_k(cumul_mod_df['x'].values, x, ensemble_knots)
    refit_spline_options = ln_daily_spline_options.copy()
    
    # run refit
    _combiner = functools.partial(run_smoothing_model,
                                  n_i_knots=n_i_knots,
                                  spline_options=refit_spline_options,
                                  pred_df=refit_pred_df,
                                  ensemble_knots=rescaled_ensemble_knots,
                                  results_only=True,
                                  log=True)
    with multiprocessing.Pool(int(F_THREAD)) as p:
        smooth_draws = list(tqdm.tqdm(p.imap(_combiner, draw_mod_dfs), total=n_draws))
    smooth_draws = np.vstack(smooth_draws).T

    # make pretty (in linear cumulative space)
    noisy_draws = draw_cleanup(noisy_draws,
                               x_pred, df)
    smooth_draws = draw_cleanup(smooth_draws,
                                x_pred, df)
    
    # get best knots and betas
    best_settings = find_best_settings(cumul_model, ln_daily_spline_options)

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
