import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from mr_spline import SplineFit


def apply_floor(vals: np.array, floor_val: float) -> np.array:
    vals[vals < floor_val] = floor_val
    
    return vals


def smoother(df: pd.DataFrame, obs_var: str, pred_var: str, 
             n_draws: int, daily: bool, log: bool) -> pd.DataFrame:
    # extract inputs
    df = df.sort_values('Date').reset_index(drop=True)
    floor = 0.01 / df['population'][0]
    keep_idx = ~df[[obs_var, pred_var]].isnull().all(axis=1)
    no_na_idx = ~df[[obs_var, pred_var]].isnull().any(axis=1)
    y = df.loc[keep_idx, [obs_var, pred_var]].values
    if daily:
        y[1:] = np.diff(y, axis=0)
    if log:
        y = apply_floor(y, floor)
        y = np.log(y)
    x = df.index[keep_idx].values

    if y[~np.isnan(y)].ptp() > 1e-10:
        # get smoothed curve (dropping NAs, inflating variance for deaths from cases - ASSUMES THAT IS SECOND COLUMN)
        obs_data = y.copy()
        obs_data[:,0] = 1
        obs_data[:,1] = 0
        y_fit = y.flatten()
        obs_data = obs_data.flatten()
        x_fit = np.repeat(x, y.shape[1], axis=0)
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
                'spline_knots_type': 'domain',
                'spline_degree': 3,
                'spline_r_linear':True,
                'spline_l_linear':True
            }
        if not daily:
            spline_options.update({'prior_spline_monotonicity':'increasing'})
        mr_mod = SplineFit(
            data=mod_df, 
            dep_var='y',
            spline_var='x',
            indep_vars=['intercept'], 
            n_i_knots=4,
            spline_options=spline_options,
            scale_se=daily,
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

def synthesize_time_series(df: pd.DataFrame, 
                           #daily: bool, log: bool, 
                           dep_var: str, spline_var: str, indep_vars: List[str],
                           n_draws: int = 1000, plot_dir: str =None) -> pd.DataFrame:
    # spline on output (first determine space based on number of deaths)
    log = True
    if (df['Death rate'] * df['population']).max() < 20:
        daily = False
    else:
        daily = True
    draw_df = smoother(df.copy().reset_index(drop=True), dep_var, f'Predicted {dep_var.lower()}', n_draws, daily, log)
    draw_cols = [col for col in draw_df.columns if col.startswith('draw_')]
    
    # add summary stats to dataset for plotting
    df = df.sort_values('Date').set_index('Date')
    draw_df = draw_df.sort_values('Date').set_index('Date')
    df['Smoothed predicted death rate'] = np.mean(draw_df[draw_cols], axis=1)
    df['Smoothed predicted death rate lower'] = np.percentile(draw_df[draw_cols], 2.5, axis=1)
    df['Smoothed predicted death rate upper'] = np.percentile(draw_df[draw_cols], 97.5, axis=1)
    df['Smoothed predicted daily death rate'] = np.nan
    df['Smoothed predicted daily death rate'][1:] = np.mean(np.diff(draw_df[draw_cols], axis=0), 
                                                            axis=1)
    df['Smoothed predicted daily death rate lower'] = np.nan
    df['Smoothed predicted daily death rate lower'][1:] = np.percentile(np.diff(draw_df[draw_cols], axis=0), 
                                                                        2.5, axis=1)
    df['Smoothed predicted daily death rate upper'] = np.nan
    df['Smoothed predicted daily death rate upper'][1:] = np.percentile(np.diff(draw_df[draw_cols], axis=0), 
                                                                        97.5, axis=1)
    df = df.reset_index()
    draw_df = draw_df.reset_index()
    first_day = df['Date'] == df.groupby('location_id')['Date'].transform(min)
    df.loc[first_day, 'Smoothed predicted daily death rate'] = df['Smoothed predicted death rate']
    df.loc[first_day, 'Smoothed predicted daily death rate lower'] = df['Smoothed predicted death rate lower']
    df.loc[first_day, 'Smoothed predicted daily death rate upper'] = df['Smoothed predicted death rate upper']
    
    # format draw data for infectionator
    draw_df = draw_df.rename(index=str, columns={'Date':'date'})
    draw_df[draw_cols] = draw_df[draw_cols] * draw_df[['population']].values
    del draw_df['population']
    
    # plot
    if plot_dir is not None:
        plotter(df, 
                [dep_var, spline_var] + indep_vars,
                f"{plot_dir}/{df['location_id'][0]}.pdf")
    
    return draw_df


def plotter(df: pd.DataFrame, unadj_vars: List[str], plot_file: str):
    # set up plot
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(2, len(unadj_vars), figsize=(len(unadj_vars)*11, 16))

    # aesthetic features
    raw_lines = {'color':'navy', 'alpha':0.5, 'linewidth':3}
    raw_points = {'c':'dodgerblue', 'edgecolors':'navy', 's':100, 'alpha':0.5}
    pred_lines = {'color':'forestgreen', 'alpha':0.75, 'linewidth':3}
    smoothed_pred_lines = {'color':'firebrick', 'alpha':0.75, 'linewidth':3}
    smoothed_pred_area = {'color':'firebrick', 'alpha':0.25}

    ax[0, 1].scatter(df['Confirmed case rate'], 
                     df['Death rate'], 
                     **raw_points)
    ax[0, 1].plot(df.loc[~df['Death rate'].isnull(), 'Confirmed case rate'], 
                  df.loc[~df['Death rate'].isnull(), 'Predicted death rate'], 
                  **pred_lines)
    ax[0, 1].plot(df.loc[~df['Death rate'].isnull(), 'Confirmed case rate'], 
                  df.loc[~df['Death rate'].isnull(), 'Smoothed predicted death rate'], 
                  **smoothed_pred_lines)
    ax[0, 1].set_xlabel('Cumulative case rate', fontsize=10)
    ax[0, 1].set_ylabel('Cumulative death rate', fontsize=10)
    
    for i, smooth_variable in enumerate(unadj_vars):
        # cumulative
        raw_variable = smooth_variable.replace('Smoothed ', '').capitalize()
        plot_label = raw_variable.lower().replace(' rate', 's')
        if ~df[raw_variable].isnull().all():
            if 'death' in smooth_variable.lower():
                ax[0, i].plot(df['Date'], df[raw_variable] * df['population'], **raw_lines)
                ax[0, i].scatter(df['Date'], df[raw_variable] * df['population'], **raw_points)
                ax[1, i].set_xlabel('Date', fontsize=10)
                ax[0, i].set_ylabel(f'Cumulative {plot_label}', fontsize=10)

            # daily
            ax[1, i].plot(df['Date'][1:], 
                          np.diff(df[raw_variable]) * df['population'][1:], 
                          **raw_lines)
            ax[1, i].scatter(df['Date'][1:], 
                             np.diff(df[raw_variable]) * df['population'][1:], 
                             **raw_points)
            ax[1, i].axhline(0, color='black', alpha=0.25)
            if 'death' in smooth_variable.lower():
                ax[1, i].set_xlabel('Date', fontsize=10)
            else:
                ax[1, i].set_xlabel('Date (+8 days)', fontsize=10)
            ax[1, i].set_ylabel(f'Daily {plot_label}', fontsize=10)

    # model prediction
    ax[0, 0].plot(df['Date'], df['Predicted death rate'] * df['population'], 
                  **pred_lines)
    ax[1, 0].plot(df['Date'][1:], 
                  np.diff(df['Predicted death rate']) * df['population'][1:], 
                  **pred_lines)
    
    # smoothed
    ax[0, 0].plot(df['Date'], 
                  df['Smoothed predicted death rate'] * df['population'], 
                  **smoothed_pred_lines)
    ax[0, 0].fill_between(
        df['Date'],
        df['Smoothed predicted death rate lower'] * df['population'], 
        df['Smoothed predicted death rate upper'] * df['population'], 
        **smoothed_pred_area
    )
    ax[1, 0].plot(df['Date'], 
                  df['Smoothed predicted daily death rate'] * df['population'], 
                  **smoothed_pred_lines)
    ax[1, 0].fill_between(
        df['Date'],
        df['Smoothed predicted daily death rate lower'] * df['population'], 
        df['Smoothed predicted daily death rate upper'] * df['population'], 
        **smoothed_pred_area
    )
        
    fig.suptitle(df['location_name'].values[0], y=1.0025, fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
    