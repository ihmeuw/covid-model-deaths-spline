from typing import List

import collections
import pandas as pd

from covid_model_deaths_spline.data import fill_dates
from db_queries import get_location_metadata


# Just interested in US plots for now
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
        subdf = df[df['location_id'].isin(child_ids)].copy().assign(location_id=aggregate_id, location_name=aggregate_name)
        if not subdf.empty:
            group_cols = ['Date', 'location_name', 'location_id']
            subdf = subdf.groupby(group_cols, as_index=False).agg(lambda x: x.sum(skipna=False))
            # only keep dates with at least 95% of potential population
            max_pop = subdf['population'].max()
            subdf = subdf.loc[subdf['population'] >= max_pop * 0.95]
            subdf[rate_cols] = subdf[rate_cols].divide(df['population'], axis=0)
            subdf['population'] = max_pop
            agg_dfs.append(subdf)
    agg_df = pd.concat(agg_dfs, ignore_index=True)
    
    for rate_col in rate_cols:
        agg_df = (agg_df.groupby('location_id', as_index=False)
                  .apply(lambda x: fill_dates(x, rate_col))
                  .reset_index(drop=True))

    return agg_df


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


def get_agg_hierarchy(hierarchy: pd.DataFrame):
    hierarchy = hierarchy.copy()
        
    # attach regions
    hierarchy['top_parent'] = hierarchy['path_to_top_parent'].apply(lambda x: int(x.split(',')[0]))
    gbd_hierarchy = get_location_metadata(location_set_id=35, gbd_round_id=6)
    gbd_hierarchy = gbd_hierarchy.loc[gbd_hierarchy['level'] <= 3].reset_index(drop=True)
    gbd_hierarchy = gbd_hierarchy.rename(index=str, columns={'location_id':'top_parent'})
    hierarchy = hierarchy.merge(gbd_hierarchy[['top_parent', 'level', 'region_id', 'sort_order']], how='left')
    hierarchy['region_id'] = hierarchy['region_id'].astype(int)
    if (hierarchy['level'] != 3).any():
        raise ValueError('Non level 3 parent in hierarchy.')
    hierarchy['path_to_top_parent'] = hierarchy['region_id'].astype(str) + ',' + hierarchy['path_to_top_parent']
    hierarchy = hierarchy.drop(['top_parent', 'level', 'region_id'], axis=1)
    hierarchy['path_to_top_parent'] = '1,' + hierarchy['path_to_top_parent']
    
    # countries with subnats
    region_ids = hierarchy['path_to_top_parent'].apply(lambda x: x.split(',')[1]).unique().tolist()
    country_ids = (hierarchy['path_to_top_parent']
                   .apply(lambda x: x.split(',')[2] if len(x.split(',')) > 3 else '')
                   .unique())
    country_ids = country_ids[country_ids != ''].tolist()
    agg_location_ids = ['1'] + sorted(country_ids) + sorted(region_ids)
    agg_locations = []
    for agg_location_id in agg_location_ids:
        agg_locations.append(Location(location_id=int(agg_location_id), 
                                      location_name=gbd_hierarchy.loc[gbd_hierarchy['top_parent'] == int(agg_location_id),
                                                                      'location_name'].item()))
    
    return hierarchy, agg_locations


def get_sorted_hierarchy_w_aggs(hierarchy: pd.DataFrame, agg_locations: collections.namedtuple) -> pd.DataFrame:
    plot_hierarchy = hierarchy.sort_values(['sort_order', 'location_id']).reset_index(drop=True)
    for agg_location_id, agg_location_name in agg_locations:
        sort_order = plot_hierarchy.loc[plot_hierarchy['path_to_top_parent']
                                        .apply(lambda x: str(agg_location_id) in x.split(',')), 
                                        'sort_order'].min()
        sort_order -= 0.01
        plot_hierarchy = (plot_hierarchy
                          .append(pd.DataFrame({'location_id': agg_location_id,
                                                'location_name': agg_location_name,
                                                'path_to_top_parent': '',
                                                'sort_order': sort_order},
                                               index=[0])))
    plot_hierarchy.loc[plot_hierarchy['location_id'] == 1, 'sort_order'] = 0
    
    
    plot_hierarchy_neg = plot_hierarchy.copy()
    plot_hierarchy_neg['location_id'] = -plot_hierarchy_neg['location_id']
    plot_hierarchy = plot_hierarchy.append(plot_hierarchy_neg)
    plot_hierarchy = plot_hierarchy.sort_values(['sort_order', 'location_id']).reset_index(drop=True)
    
    return plot_hierarchy
