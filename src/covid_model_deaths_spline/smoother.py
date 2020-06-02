import functools
from itertools import compress
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import tqdm

from covid_model_deaths_spline.mr_spline import SplineFit
from covid_model_deaths_spline.plotter import plotter


def apply_floor(vals: np.array, floor_val: float) -> np.array:
    vals[vals < floor_val] = floor_val

    return vals


def run_smoothing_model(mod_df: pd.DataFrame, n_i_knots: int, spline_options: Dict,
                        pred_df: pd.DataFrame, ensemble_knots: np.array = None) -> np.array:
        mr_model = SplineFit(
            data=mod_df,
            dep_var='y',
            spline_var='x',
            indep_vars=['intercept'],
            n_i_knots=n_i_knots,
            spline_options=spline_options,
            ensemble_knots=ensemble_knots,
            scale_se=True,
            observed_var='observed',
            pseudo_se_multiplier=1.33
        )
        mr_model.fit_model()
        smooth_y = mr_model.predict(pred_df)
        
        if ensemble_knots is None:
            return smooth_y, mr_model.mr_model.ensemble_knots
        else:
            return smooth_y
    
    
def draw_cleanup(draws: np.array, log: bool, daily: bool, smooth_y: np.array, x: np.array,
                 df: pd.DataFrame) -> pd.DataFrame:
    # set to linear, add up cumulative, and create dataframe
    if log:
        draws = np.exp(draws)
        draws *= np.exp(smooth_y) / draws.mean(axis=1, keepdims=True)
    if daily:
        draws = draws.cumsum(axis=0)

    # store in dataframe
    draw_df = df.loc[x, ['location_id', 'Date', 'population']].reset_index(drop=True)
    draw_df['Smooth log'] = log
    draw_df['Smooth daily'] = daily
    draw_df = pd.concat([draw_df, 
                         pd.DataFrame(draws, columns=[f'draw_{d}' for d in range(draws.shape[1])])], axis=1)
    
    return draw_df


def smoother(df: pd.DataFrame, obs_var: str, pred_vars: List[str],
             n_draws: int, daily: bool, log: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # extract inputs
    df = df.sort_values('Date').reset_index(drop=True)
    floor = 0.01 / df['population'][0]
    keep_idx = ~df[[obs_var] + pred_vars].isnull().all(axis=1)
    max_1week_of_zeros = (df[obs_var][::-1] == 0).cumsum()[::-1] <= 7
    y = df.loc[keep_idx, [obs_var] + pred_vars].values
    if daily:
        y[1:] = np.diff(y, axis=0)
    if log:
        y = apply_floor(y, floor)
        y = np.log(y)
    x = df.index[keep_idx].values

    # get smoothed curve (dropping NAs, inflating variance for pseudo-deaths)
    obs_data = y[max_1week_of_zeros].copy()
    obs_data[:,0] = 1
    obs_data[:,1:] = 0
    y_fit = y[max_1week_of_zeros].flatten()
    obs_data = obs_data.flatten()
    x_fit = np.repeat(x[max_1week_of_zeros], y.shape[1], axis=0)
    non_na_idx = ~np.isnan(y_fit)
    y_fit = y_fit[non_na_idx]
    obs_data = obs_data[non_na_idx]
    x_fit = x_fit[non_na_idx]
    mod_df = pd.DataFrame({
        'y':y_fit,
        'intercept':1,
        'x':x_fit,
        'observed':obs_data
    })
    mod_df['observed'] = mod_df['observed'].astype(bool)
    spline_options={
            'spline_knots_type': 'frequency',
            'spline_degree': 3,
            'spline_r_linear':True,
            'spline_l_linear':True,
        }
    if not daily:
        spline_options.update({'prior_spline_monotonicity':'increasing'})
        n_i_knots = 4
    else:
        n_i_knots = 5
    pred_df = pd.DataFrame({'intercept':1, 'x': x})
    smooth_y, ensemble_knots = run_smoothing_model(mod_df, n_i_knots, spline_options, pred_df)

    # get uncertainty in ln(daily)
    if log:
        y = np.exp(y)
        smooth_y = np.exp(smooth_y)
    if not daily:
        y[1:] = np.diff(y, axis=0)
        smooth_y[1:] = np.diff(smooth_y)
    if not log or not daily:
        y = apply_floor(y, floor)
        smooth_y = apply_floor(smooth_y, floor)
    y = np.log(y)
    smooth_y = np.log(smooth_y)
    smooth_y = np.array([smooth_y]).T
    residuals = y - smooth_y
    residuals = residuals[~np.isnan(residuals)]
    mad = np.median(np.abs(residuals))
    rstd = mad * 1.4826
    noisy_draws = np.random.normal(smooth_y, rstd, (smooth_y.size, n_draws))
    #draws = stats.t.rvs(dof, loc=smooth_y, scale=std, size=(smooth_y.size, n_draws))
    
    # refit - currently in ln(daily), transform to correct fit space
    noisy_draws = np.exp(noisy_draws)
    smooth_y = np.exp(smooth_y)
    if not daily:
        noisy_draws = np.cumsum(noisy_draws, axis=0)
        smooth_y = np.cumsum(smooth_y, axis=0)
    if log:
        noisy_draws = np.log(noisy_draws)
        smooth_y = np.log(smooth_y)
    draw_mod_dfs = [
        pd.DataFrame({
            'y':nd,
            'intercept':1,
            'x':x,
            'observed':True
        }) for nd in noisy_draws.T]
    
    rescale_k = lambda x1, k, x2: (np.quantile(x1, k) - x2.min()) / x2.ptp()
    scaled_ensemble_knots = [rescale_k(x_fit, ek, x) for ek in ensemble_knots]
    scaled_ensemble_knots = np.vstack(scaled_ensemble_knots)
    _combiner = functools.partial(run_smoothing_model,
                                  n_i_knots=n_i_knots,
                                  spline_options=spline_options,
                                  pred_df=pred_df,
                                  ensemble_knots=scaled_ensemble_knots)
    with multiprocessing.Pool(20) as p:
        smooth_draws = list(tqdm.tqdm(p.imap(_combiner, draw_mod_dfs), total=n_draws))
    smooth_draws = np.vstack(smooth_draws).T
    
    # make pretty (in linear cumulative space)
    noisy_draws = draw_cleanup(noisy_draws, log, daily, smooth_y, x, df)
    smooth_draws = draw_cleanup(smooth_draws, log, daily, smooth_y, x, df)

    return noisy_draws, smooth_draws


def refit_parallel(data: List[pd.DataFrame],
                   **model_args) -> pd.DataFrame:
    _combiner = functools.partial(synthesize_time_series,
                                  data=data, plot_dir=plot_dir,
                                  **model_args)
    location_ids = data['location_id'].unique().tolist()
    with multiprocessing.Pool(20) as p:
        draw_data_dfs = list(tqdm.tqdm(p.imap(_combiner, location_ids), total=len(location_ids)))
    return pd.concat(draw_data_dfs).reset_index(drop=True)


def synthesize_time_series(location_id: int,
                           data: pd.DataFrame,
                           obs_var: str, pred_vars: List[str],
                           spline_vars: List[str],
                           n_draws: int = 1000, plot_dir: str = None) -> pd.DataFrame:
    # location data
    df = data[data.location_id == location_id]

    # spline on output (first determine space based on number of deaths)
    log = True
    if (df['Death rate'] * df['population']).max() < 20:
        daily = False
    else:
        daily = True
    noisy_draws, smooth_draws = smoother(
        df=df.copy(), 
        obs_var=obs_var, 
        pred_vars=pred_vars, 
        n_draws=n_draws, 
        daily=daily, 
        log=log
    )
    draw_cols = [col for col in noisy_draws.columns if col.startswith('draw_')]

    # add summary stats to dataset for plotting
    summ_df = smooth_draws.copy()
    summ_df = summ_df.sort_values('Date')
    summ_df['Smoothed predicted death rate'] = np.mean(summ_df[draw_cols], axis=1)
    summ_df['Smoothed predicted death rate lower'] = np.percentile(summ_df[draw_cols], 2.5, axis=1)
    summ_df['Smoothed predicted death rate upper'] = np.percentile(summ_df[draw_cols], 97.5, axis=1)
    summ_df['Smoothed predicted daily death rate'] = np.nan
    summ_df['Smoothed predicted daily death rate'][1:] = np.mean(np.diff(summ_df[draw_cols], axis=0),
                                                                 axis=1)
    summ_df['Smoothed predicted daily death rate lower'] = np.nan
    summ_df['Smoothed predicted daily death rate lower'][1:] = np.percentile(np.diff(summ_df[draw_cols], axis=0),
                                                                             2.5, axis=1)
    summ_df['Smoothed predicted daily death rate upper'] = np.nan
    summ_df['Smoothed predicted daily death rate upper'][1:] = np.percentile(np.diff(summ_df[draw_cols], axis=0),
                                                                             97.5, axis=1)
    summ_df = summ_df[['Date'] + [i for i in summ_df.columns if i.startswith('Smoothed predicted')]]

    first_day = summ_df['Date'] == summ_df['Date'].min()
    summ_df.loc[first_day, 'Smoothed predicted daily death rate'] = summ_df['Smoothed predicted death rate']
    summ_df.loc[first_day, 'Smoothed predicted daily death rate lower'] = summ_df['Smoothed predicted death rate lower']
    summ_df.loc[first_day, 'Smoothed predicted daily death rate upper'] = summ_df['Smoothed predicted death rate upper']
    df = df.merge(summ_df, how='left')
    df = df.sort_values('Date')

    # format draw data for infectionator
    noisy_draws = noisy_draws.rename(index=str, columns={'Date':'date'})
    smooth_draws = smooth_draws.rename(index=str, columns={'Date':'date'})
    noisy_draws[draw_cols] = noisy_draws[draw_cols] * noisy_draws[['population']].values
    smooth_draws[draw_cols] = smooth_draws[draw_cols] * smooth_draws[['population']].values
    del noisy_draws['population']
    del smooth_draws['population']

    # plot
    if plot_dir is not None:
        plotter(df,
                [obs_var] + list(compress(spline_vars, (~df[spline_vars].isnull().all(axis=0)).to_list())),
                f"{plot_dir}/{df['location_id'][0]}.pdf")

    return noisy_draws, smooth_draws
