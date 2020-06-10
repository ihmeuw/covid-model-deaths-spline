from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

def get_plot_idx(i: int, n_vars: int):
    if n_vars > 1:
        top_idx = 0, i
        bottom_idx = 1, i
    else:
        top_idx = 0
        bottom_idx = 1
    return top_idx, bottom_idx


def plotter(df: pd.DataFrame, plot_vars: List[str], draw_df: pd.DataFrame, plot_file: str = None):
    # set up plot
    sns.set_style('whitegrid')
    n_cols = len(plot_vars)
    n_rows = 3
    widths = [1] * n_cols
    if n_cols < 3:
        heights = [1, 1, 1]
    else:
        heights = [1, 1, 1.5]
    fig = plt.figure(figsize=(n_cols*11, 24), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols, width_ratios=widths, height_ratios=heights)

    # aesthetic features
    raw_points = {'c':'dodgerblue', 'edgecolors':'navy', 's':100, 'alpha':0.75}
    raw_lines = {'color':'navy', 'alpha':0.5, 'linewidth':3}
    cfr_lines = {'color':'forestgreen', 'alpha':0.5, 'linewidth':3}
    hfr_lines = {'color':'darkorchid', 'alpha':0.5, 'linewidth':3}
    smoothed_pred_lines = {'color':'firebrick', 'alpha':0.75, 'linewidth':3}
    smoothed_pred_area = {'color':'firebrick', 'alpha':0.25}

    # cases
    indep_idx = 1
    if 'Confirmed case rate' in plot_vars:
        #plt.subplot(int(f'{n_rows}{n_cols}{indep_idx}'))
        ax_cfr = fig.add_subplot(gs[0, indep_idx])
        ax_cfr.scatter(df['Confirmed case rate'],
                                 df['Death rate'],
                                 **raw_points)
        ax_cfr.plot(df['Confirmed case rate'],
                              df['Predicted death rate (CFR)'],
                              **cfr_lines)
        ax_cfr.plot(df['Confirmed case rate'],
                              df['Smoothed predicted death rate'],
                              **smoothed_pred_lines)    
        ax_cfr.set_xlabel('Cumulative case rate', fontsize=14)
        ax_cfr.set_ylabel('Cumulative death rate', fontsize=14)
        indep_idx += 1

    # hospitalizations
    if 'Hospitalization rate' in plot_vars:
        #plt.subplot(int(f'{n_rows}{n_cols}{indep_idx}'))
        ax_hfr = fig.add_subplot(gs[0, indep_idx])
        ax_hfr.scatter(df['Hospitalization rate'],
                                 df['Death rate'],
                                 **raw_points)
        ax_hfr.plot(df['Hospitalization rate'],
                              df['Predicted death rate (HFR)'],
                              **hfr_lines)
        ax_hfr.plot(df['Hospitalization rate'],
                              df['Smoothed predicted death rate'],
                              **smoothed_pred_lines)
        ax_hfr.set_xlabel('Cumulative hospitalization rate', fontsize=14)
        ax_hfr.set_ylabel('Cumulative death rate', fontsize=14)
        
    for i, smooth_variable in enumerate(plot_vars):
        # get index
        top_idx, bottom_idx = get_plot_idx(i, n_cols)
        
        # cumulative
        raw_variable = smooth_variable.replace('Smoothed ', '').capitalize()
        plot_label = raw_variable.lower().replace(' rate', 's')
        if ~df[raw_variable].isnull().all():
            if 'death' in smooth_variable.lower():
                #plt.subplot(int(f'{n_rows}{n_cols}{i + 1}'))
                ax_cumul = fig.add_subplot(gs[top_idx])
                ax_cumul.plot(df['Date'], df[raw_variable] * df['population'], **raw_lines)
                ax_cumul.scatter(df['Date'], df[raw_variable] * df['population'], **raw_points)
                ax_cumul.set_ylabel(f'Cumulative {plot_label}', fontsize=14)
                ax_cumul.set_xlabel('Date', fontsize=14)


            # daily
            #plt.subplot(int(f'{n_rows}{n_cols}{i + 1 + n_cols}'))
            ax_daily = fig.add_subplot(gs[bottom_idx])
            ax_daily.plot(df['Date'][1:],
                                np.diff(df[raw_variable]) * df['population'][1:],
                                **raw_lines)
            ax_daily.scatter(df['Date'][1:],
                                   np.diff(df[raw_variable]) * df['population'][1:],
                                   **raw_points)
            ax_daily.axhline(0, color='black', alpha=0.5)
            if 'death' in smooth_variable.lower():
                ax_daily.set_xlabel('Date', fontsize=14)
            else:
                ax_daily.set_xlabel('Date (+8 days)', fontsize=14)
            ax_daily.set_ylabel(f'Daily {plot_label}', fontsize=14)
    
    # predictions - cumul
    #plt.subplot(int(f'{n_rows}{n_cols}{1}'))
    ax_cumul = fig.add_subplot(gs[0, 0])
    ax_cumul.plot(df['Date'], df['Predicted death rate (CFR)'] * df['population'],
                     **cfr_lines)
    ax_cumul.plot(df['Date'], df['Predicted death rate (HFR)'] * df['population'],
                     **hfr_lines)
    ax_cumul.plot(df['Date'],
                 df['Smoothed predicted death rate'] * df['population'],
                 **smoothed_pred_lines)
    ax_cumul.fill_between(
        df['Date'],
        df['Smoothed predicted death rate lower'] * df['population'],
        df['Smoothed predicted death rate upper'] * df['population'],
        **smoothed_pred_area
    )
    
    # predictions - dailt
    #plt.subplot(int(f'{n_rows}{n_cols}{1+n_cols}'))
    ax_daily = fig.add_subplot(gs[1, 0])
    ax_daily.plot(df['Date'][1:],
                        np.diff(df['Predicted death rate (CFR)']) * df['population'][1:],
                        **cfr_lines)
    ax_daily.plot(df['Date'][1:],
                        np.diff(df['Predicted death rate (HFR)']) * df['population'][1:],
                        **hfr_lines)
    ax_daily.plot(df['Date'],
                        df['Smoothed predicted daily death rate'] * df['population'],
                        **smoothed_pred_lines)
    ax_daily.fill_between(
        df['Date'],
        df['Smoothed predicted daily death rate lower'] * df['population'],
        df['Smoothed predicted daily death rate upper'] * df['population'],
        **smoothed_pred_area
    )
    
    # smoothed draws - ln(rate)
    draw_df = draw_df.copy()
    draw_cols = [col for col in draw_df.columns if col.startswith('draw_')]
    draw_df.iloc[1:][draw_cols] = np.diff(draw_df[draw_cols], axis=0)
    draw_df[draw_cols] = np.log(draw_df[draw_cols])
    df = df.copy()
    df['Death rate'][1:] = np.diff(df['Death rate'])
    floor = 0.01 / df['population'].values[0]
    df.loc[df['Death rate'] < floor, 'Death rate'] = floor
    #plt.subplot(int(f'{n_rows}{1}{n_rows}'))
    ax_draws = fig.add_subplot(gs[2:, 0:])
    ax_draws.plot(draw_df['Date'],
             draw_df[draw_cols],
             color='firebrick', alpha=0.025)
    ax_draws.plot(draw_df['Date'],
             draw_df[draw_cols].mean(axis=1),
             color='firebrick', linestyle='--', linewidth=3)
    ax_draws.set_ylabel('ln(daily death rate)', fontsize=14)
    ax_draws.set_xlabel('Date', fontsize=14)
    ax_draws.plot(df['Date'],
                  np.log(df['Death rate']),
                        **raw_lines)
    ax_draws.scatter(df['Date'],
                     np.log(df['Death rate']),
                     **raw_points)

    fig.suptitle(df['location_name'].values[0], y=1.0025, fontsize=24)
    fig.tight_layout()
    if plot_file:
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
