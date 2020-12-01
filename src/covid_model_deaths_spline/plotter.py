from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

DOY_COLORS = {
    'Sunday':'grey',
    'Monday':'indianred',
    'Tuesday':'peru',
    'Wednesday':'olive',
    'Thursday':'seagreen',
    'Friday':'royalblue',
    'Saturday':'darkmagenta'
}


def get_plot_idx(i: int, n_vars: int):
    if n_vars > 1:
        top_idx = 0, i
        bottom_idx = 1, i
    else:
        top_idx = 0
        bottom_idx = 1
    return top_idx, bottom_idx


def plotter(df: pd.DataFrame, plot_vars: List[str], draw_df: pd.DataFrame,
            model_labels: List[str], draw_ranges: List[Tuple[int, int]],
            plot_file: str = None):
    # set up plot
    sns.set_style('whitegrid')
    n_cols = max(len(plot_vars), 1)
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
                ax_cumul = fig.add_subplot(gs[top_idx])
                ax_cumul.plot(df['Date'], df[raw_variable] * df['population'], **raw_lines)
                ax_cumul.scatter(df['Date'], df[raw_variable] * df['population'], **raw_points)
                ax_cumul.set_ylabel(f'Cumulative {plot_label}', fontsize=14)
                ax_cumul.set_xlabel('Date', fontsize=14)


            # daily
            ax_daily = fig.add_subplot(gs[bottom_idx])
            ax_daily.plot(df['Date'],
                          np.diff(df[raw_variable], prepend=0) * df['population'],
                          **raw_lines)
            ax_daily.scatter(df['Date'],
                             np.diff(df[raw_variable], prepend=0) * df['population'],
                             **raw_points)
            ax_daily.axhline(0, color='black', alpha=0.5)
            if 'death' in smooth_variable.lower():
                ax_daily.set_xlabel('Date', fontsize=14)
            else:
                ax_daily.set_xlabel('Date of death', fontsize=14)
            ax_daily.set_ylabel(f'Daily {plot_label}', fontsize=14)

    # predictions - cumul
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

    # predictions - daily
    ax_daily = fig.add_subplot(gs[1, 0])
    ax_daily.plot(df['Date'],
                  np.diff(df['Predicted death rate (CFR)'], prepend=0) * df['population'],
                  **cfr_lines)
    ax_daily.plot(df['Date'],
                  np.diff(df['Predicted death rate (HFR)'], prepend=0) * df['population'],
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

    ## smoothed draws - ln(rate)
    # floor
    floor = 0.05 / df['population'].values[0]

    # format draws
    draw_df = draw_df.copy()
    draw_cols = [col for col in draw_df.columns if col.startswith('draw_')]
    draw_data = np.diff(draw_df[draw_cols], axis=0, prepend=0)
    draw_df[draw_cols] = draw_data

    # format model inputs
    df = df.copy()

    for input_var in ['Death rate', 'Predicted death rate (CFR)', 'Predicted death rate (HFR)']:
        df[input_var][1:] = np.diff(df[input_var])
        df.loc[df[input_var] < floor, input_var] = floor

    # plot draws
    show_draws = np.arange(0, len(draw_cols), 10).tolist()
    show_draws += [len(draw_cols) - 1]
    ax_draws = fig.add_subplot(gs[2:, 0:])
    for model_label, draw_range in zip(model_labels, draw_ranges):
        # which day
        if len(model_labels) > 1:
            doy = model_label[model_label.find('(') + 1:model_label.find(')')]
            color = DOY_COLORS[doy]
        else:
            color = 'firebrick'
        # submodel draws
        ax_draws.plot(draw_df['Date'],
                      np.log(draw_df[[f'draw_{d}' for d in range(*draw_range) if d in show_draws]]),
                      color=color, alpha=0.1)
        # submodel means
        ax_draws.plot(draw_df['Date'],
                      np.log(draw_df[[f'draw_{d}' for d in range(*draw_range)]]).mean(axis=1),
                      color=color, linewidth=3, label=model_label)
    # overall mean
    ax_draws.plot(draw_df['Date'],
                  np.log(draw_df[draw_cols]).mean(axis=1),
                  color='black', linestyle='--', linewidth=3)
    ax_draws.set_ylabel('ln(daily death rate)', fontsize=18)
    ax_draws.legend(loc=2, ncol=1, fontsize=16)
    ax_draws.set_xlabel('Date', fontsize=14)

    # plot data
    if any(~df['Death rate'].isnull()):
        ax_draws.plot(df['Date'],
                      np.log(df['Death rate']),
                      **raw_lines)
        ax_draws.scatter(df['Date'],
                         np.log(df['Death rate']),
                         **raw_points)
    ax_draws.plot(df['Date'],
                  np.log(df['Predicted death rate (CFR)']),
                  **cfr_lines)
    ax_draws.plot(df['Date'],
                  np.log(df['Predicted death rate (HFR)']),
                  **hfr_lines)
    ##

    location_name = df.loc[~df['location_name'].isnull(), 'location_name'].values
    location_id = int(df.loc[~df['location_id'].isnull(), 'location_id'].values[0])
    if location_name.size > 0:
        location_name = location_name[0]
        plot_label = f'{location_name} [{location_id}]'
    else:
        plot_label = str(location_id)
    fig.suptitle(plot_label, y=1.0025, fontsize=24)
    fig.tight_layout()
    if plot_file:
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

        
def infection_plots(infections: pd.DataFrame, model_data: pd.DataFrame, 
                    ratios: pd.DataFrame, draw_cols: List[str], plot_file: str = None):
    population = model_data['population'][0]
    location_name = model_data['location_name'][0]
    location_id = model_data['location_id'][0]
    cumulative_infections = (infections
                             .rename({'Date':'date'})
                             .set_index('date')
                             .sort_index()
                             .loc[:, draw_cols]
                             .mean(axis=1)
                             .rename('daily_infections')
                             .dropna())
    daily_infections = (infections
                        .rename({'Date':'date'})
                        .set_index('date')
                        .sort_index()
                        .loc[:, draw_cols]
                        .mean(axis=1)
                        .rename('daily_infections')
                        .diff()
                        .dropna())
    cumulative_infections *= population
    daily_infections *= population

    cumulative_cases = (model_data
                        .rename(columns={'Date':'date'})
                        .set_index('date')
                        .sort_index()
                        .loc[:, 'Confirmed case rate']
                        .rename('cases')
                        .dropna())
    daily_cases = (model_data
                   .rename(columns={'Date':'date'})
                   .set_index('date')
                   .sort_index()
                   .loc[:, 'Confirmed case rate']
                   .rename('cases')
                   .diff()
                   .dropna())
    cumulative_cases *= population
    daily_cases *= population

    ifr = (ratios
           .loc[ratios['adj_ifr'].notnull()]
           .set_index(['date'])
           .sort_index()
           .loc[:, 'ifr'])
    raw_adj_ifr = (ratios
                   .set_index(['date'])
                   .sort_index()
                   .loc[:, 'raw_adj_ifr'])
    adj_ifr = (ratios
               .set_index(['date'])
               .sort_index()
               .loc[:, 'adj_ifr'])
    cfr = (ratios
           .set_index(['date'])
           .sort_index()
           .loc[:, 'cfr'])

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(daily_infections,
                  color='coral', label='Infections')
    ax[0, 0].scatter(daily_cases.index, daily_cases,
                     c='c', edgecolors='darkcyan', label='Confirmed cases')
    ax[0, 0].set_ylabel('Daily')
    ax[0, 0].tick_params('x', labelrotation=60)

    ax[0, 1].plot(cumulative_infections,
                  color='coral', label='Infections')
    ax[0, 1].scatter(cumulative_cases.index, cumulative_cases,
                     c='c', edgecolors='darkcyan', label='Confirmed cases')
    ax[0, 1].set_ylabel('Cumulative')
    ax[0, 1].tick_params('x', labelrotation=60)

    ax[1, 0].plot(ifr, color='indianred', label='IFR')
    ax[1, 0].plot(raw_adj_ifr, linestyle='--', color='dodgerblue', label='IFR (adjusted)', alpha=0.5)
    ax[1, 0].plot(adj_ifr, color='dodgerblue', label='IFR (adjusted + smoothed)')
    ax[1, 0].set_ylabel('Infection-fatality ratio')
    ax[1, 0].tick_params('x', labelrotation=60)

    ax[1, 1].plot((daily_cases / daily_infections).rolling(window=7, min_periods=7, center=True).mean(), 
                  color='darkorchid')
    ax[1, 1].set_ylabel('Infection-detection rate (7-day average)')
    ax[1, 1].tick_params('x', labelrotation=60)

    ax[0, 0].legend(loc=2)
    ax[0, 1].legend(loc=2)
    ax[1, 0].legend(loc=1)
    
    fig.suptitle(f'{location_name} [{location_id}]', y=1.0025)
    fig.tight_layout()
    if plot_file:
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

