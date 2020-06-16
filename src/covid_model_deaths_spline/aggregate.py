from typing import List

import functools
import collections
import pandas as pd
import numpy as np

from covid_model_deaths_spline.data import fill_dates

# agg location template
Location = collections.namedtuple("Location", "location_id location_name")


def compute_location_aggregates_draws(df: pd.DataFrame, hierarchy: pd.DataFrame, aggregates: List[Location]) -> pd.DataFrame:
    """
    Aggregate draws by parent location and Date. We assume here that draws
    are in count space, and that the child locations in the draws are
    mutually exclusive and collectively exhaustive w.r.t. to the parent location.

    If some days at the end of the time series have fewer locations, we
    filter those days during aggregation.

    Parameters
    -----------
        df: draw level dataframe, should have columns Date and location_id
        hierarchy: dataframe with location_id, path_to_top_parent
        aggregates: list of locations to produce aggregates for
    """
    agg_dfs = []
    for aggregate_id, _aggregate_name in aggregates:
        child_ids = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: str(aggregate_id) in x.split(',')), 
                                  'location_id'].tolist()
        subdf = df[df['location_id'].isin(child_ids)].copy()
        if not subdf.empty:
            subdf = _drop_last_day(subdf, len(child_ids))
            subdf = subdf.groupby(['Date'], as_index=False).sum()
            subdf['location_id'] = aggregate_id
            agg_dfs.append(subdf)
    agg_df = pd.concat(agg_dfs, ignore_index=True)
    return agg_df


def compute_location_aggregates_data(df: pd.DataFrame, hierarchy: pd.DataFrame, aggregates: List[Location],
                                     rate_cols: List[str] = ['Confirmed case rate', 'Death rate', 'Predicted death rate (CFR)',
                                                             'Hospitalization rate', 'Predicted death rate (HFR)']) -> pd.DataFrame:
    """
    Aggregate draws by parent location and Date. We assume here that model data
    columns are in rate space , and that the child locations in the draws are
    mutually exclusive and collectively exhaustive w.r.t. to the parent location.

    We propagate NaN values when aggregating (we usually assume Nan is 0)
    in order ensure we do not plot observations (ie measure-dates) that are missing
    a full complement of locations

    Parameters
    -----------
        df: model data dataframe, should have columns Date, location_id,
        'Confirmed case rate', 'Death rate', 'Predicted death rate (CFR)',
        'Hospitalization rate', 'Predicted death rate (HFR)', population
        hierarchy: dataframe with location_id, path_to_top_parent
        aggregates: list of locations to produce aggregates for
    """
    df = df.copy()

    # Temporarily convert rates to counts during aggregation
    df[rate_cols] = df[rate_cols].multiply(df['population'], axis=0)

    agg_dfs = []
    for aggregate_id, aggregate_name in aggregates:
        child_ids = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: str(aggregate_id) in x.split(',')), 
                                  'location_id'].tolist()
        subdf = df[df['location_id'].isin(child_ids)].copy()
        maxdf = subdf.groupby('location_id')[rate_cols].max()
        maxdf = maxdf[maxdf.sum(axis=1) > 0].reset_index()
        subdf = (subdf.loc[subdf['location_id'].isin(maxdf['location_id'].to_list())]
                 .assign(location_id=aggregate_id, location_name=aggregate_name))
        if not subdf.empty:
            group_cols = ['Date', 'location_name', 'location_id']
            subdf = subdf.groupby(group_cols, as_index=False)[rate_cols + ['population']].agg(lambda x: x.sum(skipna=False))
            # only keep dates with at least 95% of potential population
            max_pop = subdf['population'].max()
            subdf = subdf.loc[subdf['population'] >= max_pop * 0.95]
            subdf[rate_cols] = subdf[rate_cols].divide(df['population'], axis=0)
            subdf['population'] = max_pop
            agg_dfs.append(subdf)
    agg_df = pd.concat(agg_dfs, ignore_index=True)
    
    # make sure we still have consecutive days (interpolate missing)
    col_dfs = []
    for col in rate_cols + ['population']:
        col_df = agg_df[['location_id', 'location_name', 'Date', col]]
        col_df = col_df.loc[~col_df[col].isnull()]
        if not col_df.empty:
            col_df = (col_df.groupby('location_id', as_index=False)
                      .apply(lambda x: fill_dates(x, col))
                      .reset_index(drop=True))
            col_dfs.append(col_df)
    agg_df = functools.reduce(lambda x, y: pd.merge(x, y, how='outer'), col_dfs)
    for col in rate_cols + ['population']:
        if col not in agg_df.columns:
            agg_df[col] = np.nan
    return agg_df.sort_values(['location_id', 'Date']).reset_index(drop=True)


def _drop_last_day(df: pd.DataFrame, num_locs: int):
    """
    The last day might have fewer locations than most of the time series,
    so cumulative totals would appear negative in the plots. Let's
    drop any dates near the end of the time series that would qualify
    """
    rows_per_date = df.groupby('Date').size()
    last_date_with_correct_loc_count = rows_per_date[rows_per_date == num_locs].index.max()
    dates_to_drop = rows_per_date[rows_per_date.index > last_date_with_correct_loc_count].index
    return df[~df['Date'].isin(dates_to_drop)]
