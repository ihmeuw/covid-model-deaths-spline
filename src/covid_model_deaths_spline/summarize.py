from itertools import compress
from typing import List

import numpy as np
import pandas as pd

from covid_model_deaths_spline import smoother


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
    df = df.merge(summ_df, how='outer')
    df = df.sort_values(['location_id', 'Date'])

    return df


def summarize_and_plot(agg_df: pd.DataFrame, model_data: pd.DataFrame, plot_dir: str, obs_var: str, spline_vars: List[str]) -> pd.DataFrame:
    # draws are in count space and we need to get back to rate space before
    # plotting
    agg_df = agg_df.set_index(['Date', 'location_id'])
    model_data = model_data.set_index(['Date', 'location_id'])
    draw_cols = [col for col in agg_df.columns if col.startswith('draw_')]
    agg_df[draw_cols] = agg_df[draw_cols].divide(model_data['population'], axis=0)
    agg_df = agg_df.reset_index()
    model_data = model_data.reset_index()

    summ_df = append_summary_statistics(agg_df, model_data)

    # draws sometimes have last day dropped if they're missing full compliment
    # of locations, so we should drop those days from data too
    summ_df = summ_df[summ_df['Date'] <= agg_df['Date'].max()]
    for location_id in summ_df['location_id'].unique():
        p_summ_df = summ_df.loc[summ_df['location_id'] == location_id].reset_index(drop=True)
        p_agg_df = agg_df.loc[agg_df['location_id'] == location_id].reset_index(drop=True)
        smoother.plotter(p_summ_df,
                         [obs_var] + list(compress(spline_vars, (~p_summ_df[spline_vars].isnull().all(axis=0)).to_list())),
                         p_agg_df,
                         f"{plot_dir}/{location_id}.pdf")
