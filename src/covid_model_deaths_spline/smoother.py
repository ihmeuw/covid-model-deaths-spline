import functools
from itertools import compress
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from covid_model_deaths_spline.mr_spline import SplineFit
from covid_model_deaths_spline.utilities import run_multiprocess


def apply_floor(vals: np.array, floor_val: float) -> np.array:
    vals[vals < floor_val] = floor_val

    return vals


def smoother(df: pd.DataFrame, obs_var: str, pred_vars: List[str],
             n_draws: int, daily: bool, log: bool) -> pd.DataFrame:
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

    if y[~np.isnan(y)].ptp() > 1e-10:
        # get smoothed curve (dropping NAs, inflating variance for deaths from cases - ASSUMES THAT IS SECOND COLUMN)
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
        mr_mod = SplineFit(
            data=mod_df,
            dep_var='y',
            spline_var='x',
            indep_vars=['intercept'],
            n_i_knots=5,
            spline_options=spline_options,
            scale_se=True,
            observed_var='observed',
            pseudo_se_multiplier=1.33
        )
        mr_mod.fit_model()
        smooth_y = mr_mod.predict(pd.DataFrame({'intercept':1, 'x': x}))
    else:
        # don't smooth if no difference
        smooth_y = y

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
    draws = np.random.normal(smooth_y, rstd, (smooth_y.size, n_draws))
    #draws = stats.t.rvs(dof, loc=smooth_y, scale=std, size=(smooth_y.size, n_draws))

    # set to linear, add up cumulative, and create dataframe
    draws = np.exp(draws)
    draws *= np.exp(smooth_y) / draws.mean(axis=1, keepdims=True)
    draws = draws.cumsum(axis=0)
    draw_df = df.loc[x, ['location_id', 'Date', 'population']].reset_index(drop=True)
    draw_df['Smooth log'] = log
    draw_df['Smooth daily'] = daily
    draw_df = pd.concat([draw_df, pd.DataFrame(draws, columns=[f'draw_{d}' for d in range(n_draws)])], axis=1)

    return draw_df


def synthesize_time_series_parallel(data: pd.DataFrame,
                                    plot_dir: Path,
                                    **model_args) -> pd.DataFrame:
    _combiner = functools.partial(synthesize_time_series,
                                  data=data, plot_dir=plot_dir,
                                  **model_args)
    location_ids = data['location_id'].unique().tolist()

    draw_data_dfs = run_multiprocess(_combiner, location_ids)

    return pd.concat(draw_data_dfs).reset_index(drop=True)


def synthesize_time_series(location_id: int,
                           data: pd.DataFrame,
                           dep_var: str, indep_vars: List[str],
                           n_draws: int = 1000, plot_dir: str = None) -> pd.DataFrame:
    # location data
    df = data[data.location_id == location_id]

    # spline on output (first determine space based on number of deaths)
    log = True
    if (df['Death rate'] * df['population']).max() < 20:
        daily = False
    else:
        daily = True
    draw_df = smoother(
        df=df.copy(),
        obs_var=dep_var,
        pred_vars=[f'Predicted {dep_var.lower()} (CFR)', f'Predicted {dep_var.lower()} (HFR)'],
        n_draws=n_draws,
        daily=daily,
        log=log
    )
    draw_cols = [col for col in draw_df.columns if col.startswith('draw_')]

    # add summary stats to dataset for plotting
    summ_df = draw_df.copy()
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
    draw_df = draw_df.rename(index=str, columns={'Date':'date'})
    draw_df[draw_cols] = draw_df[draw_cols] * draw_df[['population']].values
    del draw_df['population']

    # plot
    if plot_dir is not None:
        plotter(df,
                [dep_var] + list(compress(indep_vars, (~df[indep_vars].isnull().all(axis=0)).to_list())),
                f"{plot_dir}/{df['location_id'][0]}.pdf")

    return draw_df


def get_plot_idx(i: int, n_vars: int):
    if n_vars > 1:
        top_idx = 0, i
        bottom_idx = 1, i
    else:
        top_idx = 0
        bottom_idx = 1
    return top_idx, bottom_idx


def plotter(df: pd.DataFrame, unadj_vars: List[str], plot_file: str):
    # set up plot
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(2, len(unadj_vars), figsize=(len(unadj_vars)*11, 16))

    # aesthetic features
    raw_points = {'c':'dodgerblue', 'edgecolors':'navy', 's':100, 'alpha':0.75}
    raw_lines = {'color':'navy', 'alpha':0.5, 'linewidth':3}
    cfr_lines = {'color':'forestgreen', 'alpha':0.5, 'linewidth':3}
    hfr_lines = {'color':'darkorchid', 'alpha':0.5, 'linewidth':3}
    smoothed_pred_lines = {'color':'firebrick', 'alpha':0.75, 'linewidth':3}
    smoothed_pred_area = {'color':'firebrick', 'alpha':0.25}

    # cases
    indep_idx = 1
    if 'Confirmed case rate' in unadj_vars:
        ax[0, indep_idx].scatter(df['Confirmed case rate'],
                                 df['Death rate'],
                                 **raw_points)
        ax[0, indep_idx].plot(df.loc[~df['Death rate'].isnull(), 'Confirmed case rate'],
                              df.loc[~df['Death rate'].isnull(), 'Predicted death rate (CFR)'],
                              **cfr_lines)
        ax[0, indep_idx].plot(df.loc[~df['Death rate'].isnull(), 'Confirmed case rate'],
                              df.loc[~df['Death rate'].isnull(), 'Smoothed predicted death rate'],
                              **smoothed_pred_lines)
        ax[0, indep_idx].set_xlabel('Cumulative case rate', fontsize=10)
        ax[0, indep_idx].set_ylabel('Cumulative death rate', fontsize=10)
        indep_idx += 1

    # hospitalizations
    if 'Hospitalization rate' in unadj_vars:
        ax[0, indep_idx].scatter(df['Hospitalization rate'],
                                 df['Death rate'],
                                 **raw_points)
        ax[0, indep_idx].plot(df.loc[~df['Death rate'].isnull(), 'Hospitalization rate'],
                              df.loc[~df['Death rate'].isnull(), 'Predicted death rate (HFR)'],
                              **hfr_lines)
        ax[0, indep_idx].plot(df.loc[~df['Death rate'].isnull(), 'Hospitalization rate'],
                              df.loc[~df['Death rate'].isnull(), 'Smoothed predicted death rate'],
                              **smoothed_pred_lines)
        ax[0, indep_idx].set_xlabel('Cumulative hospitalization rate', fontsize=10)
        ax[0, indep_idx].set_ylabel('Cumulative death rate', fontsize=10)

    for i, smooth_variable in enumerate(unadj_vars):
        top_idx, bottom_idx = get_plot_idx(i, len(unadj_vars))

        # cumulative
        raw_variable = smooth_variable.replace('Smoothed ', '').capitalize()
        plot_label = raw_variable.lower().replace(' rate', 's')
        if ~df[raw_variable].isnull().all():
            if 'death' in smooth_variable.lower():
                ax[top_idx].plot(df['Date'], df[raw_variable] * df['population'], **raw_lines)
                ax[top_idx].scatter(df['Date'], df[raw_variable] * df['population'], **raw_points)
                ax[bottom_idx].set_xlabel('Date', fontsize=10)
                ax[top_idx].set_ylabel(f'Cumulative {plot_label}', fontsize=10)

            # daily
            ax[bottom_idx].plot(df['Date'][1:],
                                np.diff(df[raw_variable]) * df['population'][1:],
                                **raw_lines)
            ax[bottom_idx].scatter(df['Date'][1:],
                                   np.diff(df[raw_variable]) * df['population'][1:],
                                   **raw_points)
            ax[bottom_idx].axhline(0, color='black', alpha=0.25)
            if 'death' in smooth_variable.lower():
                ax[bottom_idx].set_xlabel('Date', fontsize=10)
            else:
                ax[bottom_idx].set_xlabel('Date (+8 days)', fontsize=10)
            ax[bottom_idx].set_ylabel(f'Daily {plot_label}', fontsize=10)

    # model prediction
    top_idx, bottom_idx = get_plot_idx(0, len(unadj_vars))
    ax[top_idx].plot(df['Date'], df['Predicted death rate (CFR)'] * df['population'],
                     **cfr_lines)
    ax[top_idx].plot(df['Date'], df['Predicted death rate (HFR)'] * df['population'],
                     **hfr_lines)
    ax[bottom_idx].plot(df['Date'][1:],
                        np.diff(df['Predicted death rate (CFR)']) * df['population'][1:],
                        **cfr_lines)
    ax[bottom_idx].plot(df['Date'][1:],
                        np.diff(df['Predicted death rate (HFR)']) * df['population'][1:],
                        **hfr_lines)

    # smoothed
    ax[top_idx].plot(df['Date'],
                     df['Smoothed predicted death rate'] * df['population'],
                     **smoothed_pred_lines)
    ax[top_idx].fill_between(
        df['Date'],
        df['Smoothed predicted death rate lower'] * df['population'],
        df['Smoothed predicted death rate upper'] * df['population'],
        **smoothed_pred_area
    )
    ax[bottom_idx].plot(df['Date'],
                        df['Smoothed predicted daily death rate'] * df['population'],
                        **smoothed_pred_lines)
    ax[bottom_idx].fill_between(
        df['Date'],
        df['Smoothed predicted daily death rate lower'] * df['population'],
        df['Smoothed predicted daily death rate upper'] * df['population'],
        **smoothed_pred_area
    )

    fig.suptitle(df['location_name'].values[0], y=1.0025, fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)

