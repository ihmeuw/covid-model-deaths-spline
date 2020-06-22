from itertools import compress
from typing import List

import numpy as np
import pandas as pd

from covid_model_deaths_spline import smoother
from covid_model_deaths_spline.plotter import plotter


def append_summary_statistics(draw_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary columns to the input dataframe.

    Parameters
    ----------
        draw_df: input dataframe of smoothed draws assumed to have some number of columns
            that start with 'draw_', and 'Date'
        df: input dataframe before smoother that has summary stats appended
    """

    draw_cols = [col for col in draw_df.columns if col.startswith('draw_')]

    summ_df = draw_df.copy()
    summ_df = summ_df.sort_values(['location_id', 'Date'])
    summ_df['Smoothed predicted death rate'] = np.mean(summ_df[draw_cols], axis=1)
    summ_df['Smoothed predicted death rate lower'] = np.percentile(summ_df[draw_cols], 2.5, axis=1)
    summ_df['Smoothed predicted death rate upper'] = np.percentile(summ_df[draw_cols], 97.5, axis=1)
    summ_df['Smoothed predicted daily death rate'] = np.mean(np.diff(summ_df[draw_cols], axis=0, prepend=np.nan),
                                                             axis=1)
    summ_df['Smoothed predicted daily death rate lower'] = np.percentile(np.diff(summ_df[draw_cols], axis=0, prepend=np.nan),
                                                                         2.5, axis=1)
    summ_df['Smoothed predicted daily death rate upper'] = np.percentile(np.diff(summ_df[draw_cols], axis=0, prepend=np.nan),
                                                                         97.5, axis=1)
    summ_df = summ_df[['location_id', 'Date'] + [i for i in summ_df.columns if i.startswith('Smoothed predicted')]]

    first_day = summ_df['Date'] == summ_df.groupby('location_id')['Date'].transform(min)
    summ_df.loc[first_day, 'Smoothed predicted daily death rate'] = summ_df['Smoothed predicted death rate']
    summ_df.loc[first_day, 'Smoothed predicted daily death rate lower'] = summ_df['Smoothed predicted death rate lower']
    summ_df.loc[first_day, 'Smoothed predicted daily death rate upper'] = summ_df['Smoothed predicted death rate upper']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.merge(summ_df, how='outer')
    df = df.sort_values(['location_id', 'Date'])
    
    # deal with locations without observed data
    if df['population'].isnull().any():
        df['population'] = df.groupby('location_id')['population'].transform(max)
    df['location_name'] = df.groupby('location_id')['location_name'].transform(lambda x: x.fillna(method='bfill'))
    df = df.loc[~np.isnat(df['Date'])]
    
    return df.reset_index(drop=True)


def summarize_and_plot(agg_df: pd.DataFrame, model_data: pd.DataFrame,  
                       plot_dir: str, obs_var: str, spline_vars: List[str], 
                       pop_data: pd.DataFrame = None) -> pd.DataFrame:
    # draws are in count space and we need to get back to rate space before
    # plotting
    if pop_data is not None:
        pop_data = pop_data[['location_id', 'population']]
    else:
        pop_data = model_data[['location_id', 'population']].drop_duplicates()
    draw_cols = [col for col in agg_df.columns if col.startswith('draw_')]
    agg_df = agg_df.merge(pop_data)
    agg_df[draw_cols] = agg_df[draw_cols].divide(agg_df['population'], axis=0)
    del agg_df['population']

    summ_df = append_summary_statistics(agg_df, model_data)

    # draws sometimes have last day dropped if they're missing full compliment
    # of locations, so we should drop those days from data too
    summ_df = summ_df[summ_df['Date'] <= agg_df['Date'].max()]
    for location_id in summ_df['location_id'].unique():
        p_summ_df = summ_df.loc[summ_df['location_id'] == location_id].reset_index(drop=True)
        p_agg_df = agg_df.loc[agg_df['location_id'] == location_id].reset_index(drop=True)
        plotter(
            p_summ_df,
            list(compress([obs_var] + spline_vars, 
                          (~p_summ_df[[obs_var] + spline_vars].isnull().all(axis=0)).to_list())),
            p_agg_df,
            ['Indirect'], [(0, len(draw_cols))],
            f'{plot_dir}/{location_id}.pdf'
        )
