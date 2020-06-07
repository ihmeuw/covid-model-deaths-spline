import functools
from itertools import compress
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple
from collections import namedtuple
import dill as pickle

import numpy as np
import pandas as pd
import tqdm

from covid_model_deaths_spline import summarize
from covid_model_deaths_spline.mr_spline import SplineFit, rescale_k
from covid_model_deaths_spline.plotter import plotter


def apply_floor(vals: np.array, floor_val: float) -> np.array:
    vals[vals < floor_val] = floor_val

    return vals


def run_smoothing_model(mod_df: pd.DataFrame, n_i_knots: int, spline_options: Dict, scale_se: bool,
                        pred_df: pd.DataFrame, ensemble_knots: np.array = None, 
                        results_only: bool = False) -> np.array:
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
            pseudo_se_multiplier=1.25
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
                   mono: bool, tail_gprior: np.array = None):
    # get smoothed curve (dropping NAs, inflating variance for pseudo-deaths)
    obs_data = y[subset_idx].copy()
    obs_data[:,0] = 1
    obs_data[:,1:] = 0
    y_fit = y[subset_idx].flatten()
    obs_data = obs_data.flatten()
    col_data = np.array([col_names] * y[subset_idx].shape[0]).flatten()
    x_fit = np.repeat(x[subset_idx], y.shape[1], axis=0)
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
    spline_options={
            'spline_knots_type': 'domain',
            'spline_degree': 3,
            'spline_r_linear': True,
            'spline_l_linear': True,
            'prior_spline_funval_uniform': np.array([-np.inf, 0])
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
    #draws -= np.var(draws, axis=1, keepdims=True) / 2
    draws = np.exp(draws)
    draws *= np.exp(smooth_y) / draws.mean(axis=1, keepdims=True)
    draws[draws * df['population'].values[0] < 1e-10] = 1e-10 / df['population'].values[0]
    draws = draws.cumsum(axis=0)

    # store in dataframe
    draw_df = df.loc[x, ['location_id', 'Date', 'population']].reset_index(drop=True)
    draw_df = pd.concat([draw_df,
                         pd.DataFrame(draws, columns=[f'draw_{d}' for d in range(draws.shape[1])])], axis=1)

    return draw_df


def combine_cumul_daily(ln_cumul_smooth_y: np.array, ln_daily_smooth_y: np.array, 
                        total_deaths: float) -> np.array:
    from_cumul = np.exp(ln_cumul_smooth_y)
    from_cumul[1:] = np.diff(from_cumul, axis=0)
    from_daily = np.exp(ln_daily_smooth_y)
    d_w = min(total_deaths / 100., 1.)
    c_w = 1. - d_w
    smooth_y = from_daily * d_w + from_cumul * c_w
    
    return np.log(smooth_y)


def get_mad(df: pd.DataFrame, weighted: bool) -> float:
    residuals = (df['y'] - df['smooth_y_combined']).values
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
             n_i_knots: int, n_draws: int, total_deaths: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # extract inputs
    df = df.sort_values('Date').reset_index(drop=True)
    floor = 0.01 / df['population'][0]
    keep_idx = ~df[[obs_var] + pred_vars].isnull().all(axis=1)
    max_1week_of_zeros = (df[obs_var][::-1] == 0).cumsum()[::-1] <= 7
    cumul_y = df.loc[keep_idx, [obs_var] + pred_vars].values
    daily_y = cumul_y.copy()
    daily_y[1:] = np.diff(daily_y, axis=0)
    ln_cumul_y = np.log(apply_floor(cumul_y, floor))
    ln_daily_y = np.log(apply_floor(daily_y, floor))
    x = df.index[keep_idx].values
    
    # how many days in fit window are non-zero (use this to determine cumul/daily weights?)
    # TODO: test this out to determine potential gradient
    non_zero_data = np.diff(df['Death rate'], prepend=0) > 0
    pct_non_zero = len(df.loc[max_1week_of_zeros & non_zero_data]) / len(df.loc[max_1week_of_zeros])

    # get deaths in last week
    last_week = df.copy()
    last_week['Deaths'] = last_week['Death rate'] * last_week['population']
    last_week['Deaths'][1:] = np.diff(last_week['Deaths'])
    last_week_deaths = last_week.loc[~last_week['Deaths'].isnull()].iloc[-7:]['Deaths'].sum()
    gprior_se = max(1, last_week_deaths) / 100

    # prepare data and run daily
    pred_df = pd.DataFrame({'intercept':1, 'x': x})
    ln_daily_mod_df, ln_daily_spline_options = process_inputs(
        y=ln_daily_y, col_names=[obs_var] + pred_vars,
        x=x, n_i_knots=n_i_knots, subset_idx=max_1week_of_zeros,
        mono=False, tail_gprior=np.array([0, gprior_se])
    )
    ln_daily_smooth_y, ln_daily_model, ln_daily_mod_df = run_smoothing_model(
        ln_daily_mod_df, n_i_knots, ln_daily_spline_options, True, pred_df
    )
    ensemble_knots = ln_daily_model.ensemble_knots
    #cumul_trans = np.diff(np.log(np.cumsum(np.exp(ln_daily_smooth_y))))
    #penult_k = ln_daily_mod_df['x'].min() + ensemble_knots[0][-2] * np.ptp(ln_daily_mod_df['x'])
    #cumul_gprior_mean = cumul_trans[x[1:] >= penult_k].mean()
    
    # run cumulative, using slope after the last knot as the prior mean
    ln_cumul_mod_df, ln_cumul_spline_options = process_inputs(
        y=ln_cumul_y, col_names=[obs_var] + pred_vars,
        x=x, n_i_knots=n_i_knots, subset_idx=max_1week_of_zeros,
        mono=True, tail_gprior=np.array([0, gprior_se])  # cumul_gprior_mean
    )
    ln_cumul_smooth_y, ln_cumul_model, ln_cumul_mod_df = run_smoothing_model(
        ln_cumul_mod_df, n_i_knots, ln_cumul_spline_options, False, pred_df, ensemble_knots
    )

    # average the two in linear daily (increasing influence of daily as we get closer to 100), then log
    smooth_y = combine_cumul_daily(ln_cumul_smooth_y, ln_daily_smooth_y, total_deaths)
    input_ln_cumul_smooth_y = pd.pivot_table(ln_cumul_mod_df, index='x', columns='data_type', values='smooth_y').values
    input_ln_daily_smooth_y = pd.pivot_table(ln_daily_mod_df, index='x', columns='data_type', values='smooth_y').values
    input_ln_daily_smooth_y = combine_cumul_daily(
        input_ln_cumul_smooth_y, input_ln_daily_smooth_y, total_deaths
    )
    input_ln_daily_smooth_y = input_ln_daily_smooth_y.flatten()
    ln_daily_mod_df['smooth_y_combined'] = input_ln_daily_smooth_y[~np.isnan(input_ln_daily_smooth_y)]

    # get uncertainty in ln(daily)
    smooth_y = np.array([smooth_y]).T
    mad = get_mad(ln_daily_mod_df, weighted=True)
    rstd = mad * 1.4826
    noisy_draws = np.random.normal(smooth_y, rstd, (smooth_y.size, n_draws))
    #draws = stats.t.rvs(dof, loc=smooth_y, scale=std, size=(smooth_y.size, n_draws))

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
    rescaled_ensemble_knots = rescale_k(ln_daily_mod_df['x'].values, x, ensemble_knots)
    refit_spline_options = ln_daily_spline_options.copy()
    del refit_spline_options['prior_spline_funval_uniform']
    _combiner = functools.partial(run_smoothing_model,
                                  n_i_knots=n_i_knots,
                                  spline_options=refit_spline_options,
                                  pred_df=pred_df,
                                  scale_se=False,
                                  ensemble_knots=rescaled_ensemble_knots,
                                  results_only=True)
    with multiprocessing.Pool(20) as p:
        smooth_draws = list(tqdm.tqdm(p.imap(_combiner, draw_mod_dfs), total=n_draws))
    smooth_draws = np.vstack(smooth_draws).T

    # make pretty (in linear cumulative space)
    noisy_draws = draw_cleanup(noisy_draws, smooth_y, x, df)
    smooth_draws = draw_cleanup(smooth_draws, smooth_y, x, df)
    
    # get best knots and betas
    best_settings = find_best_settings(ln_daily_model, ln_daily_spline_options)

    return noisy_draws, smooth_draws, best_settings


def synthesize_time_series(location_id: int,
                           data: pd.DataFrame,
                           obs_var: str, pred_vars: List[str],
                           spline_vars: List[str],
                           spline_settings_dir: str,
                           n_draws: int = 1000, plot_dir: str = None) -> pd.DataFrame:
    # location data
    df = data[data.location_id == location_id]

    # spline on output (first determine space based on number of deaths)
    total_deaths = (df['Death rate'] * df['population']).max()
    if len(df) >= 30 and total_deaths > 10:
        n_i_knots = 5
    elif len(df) >= 15 and total_deaths > 5:
        n_i_knots = 4
    else:
        n_i_knots = 3
    noisy_draws, smooth_draws, best_settings = smoother(
        df=df.copy(),
        obs_var=obs_var,
        pred_vars=pred_vars,
        n_i_knots=n_i_knots,
        n_draws=n_draws,
        total_deaths=total_deaths
    )
    draw_cols = [col for col in noisy_draws.columns if col.startswith('draw_')]
    df = summarize.append_summary_statistics(smooth_draws, df)
    
    # save knots
    with open(f"{spline_settings_dir}/{df['location_id'][0]}.pkl", 'wb') as fwrite:
        pickle.dump(best_settings, fwrite, -1)

    # plot
    if plot_dir is not None:
        plotter(df,
                [obs_var] + list(compress(spline_vars, (~df[spline_vars].isnull().all(axis=0)).to_list())),
                smooth_draws,
                f"{plot_dir}/{df['location_id'][0]}.pdf")
        
    # format draw data for infectionator
    noisy_draws = noisy_draws.rename(index=str, columns={'Date':'date'})
    smooth_draws = smooth_draws.rename(index=str, columns={'Date':'date'})
    noisy_draws[draw_cols] = noisy_draws[draw_cols] * noisy_draws[['population']].values
    smooth_draws[draw_cols] = smooth_draws[draw_cols] * smooth_draws[['population']].values
    del noisy_draws['population']
    del smooth_draws['population']

    return noisy_draws, smooth_draws
