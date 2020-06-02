import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plotter(df: pd.DataFrame, plot_vars: List[str], plot_file: str):
    # set up plot
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(2, len(plot_vars), figsize=(len(plot_vars)*11, 16))

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
    if 'Hospitalization rate' in plot_vars:
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

        
    for i, smooth_variable in enumerate(plot_vars):
        top_idx, bottom_idx = get_plot_idx(i, len(plot_vars))
        
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
    top_idx, bottom_idx = get_plot_idx(0, len(plot_vars))
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
