import functools
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger
import pandas as pd
import numpy as np


def evil_doings(case_data: pd.DataFrame,
                hosp_data: pd.DataFrame,
                death_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    # Record our sins
    manipulation_metadata = {}
    drop_date = pd.Timestamp('2020-07-04')
    case_drops = [44, 541, 546, 527]
    death_drops = [44, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 
                   535, 537, 538, 539, 541, 543, 544, 545, 546, 547, 548, 549, 550, 
                   551, 553, 554, 555, 557, 558, 559, 561, 562, 564, 565, 566, 567, 
                   568, 569, 570, 572, 573, 3539, 60886, 60887]
    case_data = case_data[~(case_data.location_id.isin(case_drops) & (case_data['Date'] == drop_date))]
    death_data = death_data[~(death_data.location_id.isin(death_drops) & (death_data['Date'] == drop_date))]
    manipulation_metadata['4th_of_july_madness'] = {'deaths_dropped': {'date': '2020_07_04', 'locations': death_drops},
                                                    'cases_dropped': {'date': '2020_07_04', 'locations': case_drops}}

    new_york = 555
    spike_day = pd.Timestamp('2020-06-30')
    death_data = death_data[~((death_data.location_id == new_york) & (death_data['Date'] >= spike_day))]
    manipulation_metadata['new_york_spike'] = {'deaths_dropped': {'date': 'days after 2020-06-30', 'locations': [new_york]}}

    uk = 95
    spike_day = pd.Timestamp('2020-07-02')
    case_data = case_data[~((case_data.location_id == uk) & (case_data['Date'] >= spike_day))]
    manipulation_metadata['uk_neg_spike'] = {'cases_dropped': {'date': 'days after 2020-07-02', 'locations': [uk]}}
    return case_data, hosp_data, death_data, manipulation_metadata


def load_most_detailed_locations(inputs_root: Path) -> pd.DataFrame:
    """Loads the most detailed locations in the current modeling hierarchy."""
    location_hierarchy_path = inputs_root / 'locations' / 'modeling_hierarchy.csv'
    hierarchy = pd.read_csv(location_hierarchy_path)

    most_detailed = hierarchy['most_detailed'] == 1
    keep_columns = ['location_id', 'location_name', 'path_to_top_parent', 'sort_order']

    return hierarchy.loc[most_detailed, keep_columns].sort_values('sort_order').reset_index(drop=True)


def load_aggregate_locations(inputs_root: Path) -> pd.DataFrame:
    """Loads the parent locations in the current modeling hierarchy (except global)."""
    location_hierarchy_path = inputs_root / 'locations' / 'modeling_hierarchy.csv'
    hierarchy = pd.read_csv(location_hierarchy_path)

    aggregate = hierarchy['most_detailed'] == 0
    not_global = hierarchy['location_id'] != 1
    keep_columns = ['location_id', 'location_name', 'path_to_top_parent', 'sort_order']

    return hierarchy.loc[aggregate & not_global, keep_columns].sort_values('sort_order').reset_index(drop=True)


def load_full_data(inputs_root: Path) -> pd.DataFrame:
    """Gets all death, case, and population data."""
    full_data_path = inputs_root / 'use_at_your_own_risk' / 'full_data_extra_hospital.csv'
    data = pd.read_csv(full_data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['location_id'] = data['location_id'].astype(int)

    keep_columns = ['location_id', 'Date', 'Deaths', 'Confirmed', 'Hospitalizations', 'Death rate', 'population']
    sort_columns = ['location_id', 'Date']
    data = data.loc[:, keep_columns].sort_values(sort_columns).reset_index(drop=True)
    return data


def get_shifted_data(full_data: pd.DataFrame, count_var: str, rate_var: str, shift_size: int = 8) -> pd.DataFrame:
    """Filter and clean case data and shift into the future."""
    data = full_data.loc[:, ['location_id', 'Date', count_var, 'population']]
    data[rate_var] = data[count_var] / data['population']
    data['True date'] = data['Date']
    data['Date'] = data['Date'].apply(lambda x: x + pd.Timedelta(days=shift_size))

    non_na = ~data[rate_var].isnull()
    has_data = data.groupby('location_id')[rate_var].transform(max).astype(bool)
    keep_columns = ['location_id', 'True date', 'Date', rate_var]
    data = data.loc[non_na & has_data, keep_columns].reset_index(drop=True)

    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: fill_dates(x, rate_var))
            .reset_index(drop=True))

    return data


def get_death_data(full_data: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean death data."""
    non_na = ~full_data['Deaths'].isnull()
    keep_columns = ['location_id', 'Date', 'Deaths', 'population']
    death_df = full_data.loc[non_na, keep_columns].reset_index(drop=True)
    death_df['Death rate'] = death_df['Deaths'] / death_df['population']
    del death_df['population']

    death_df = (death_df.groupby('location_id', as_index=False)
                .apply(lambda x: fill_dates(x, 'Death rate'))
                .reset_index(drop=True))

    return death_df


def get_population_data(input_root: Path, hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean population data."""
    pop_df = pd.read_csv(input_root / 'output_measures' / 'population' / 'all_populations.csv')
    pop_df = pop_df[(pop_df.age_group_id == 22) & (pop_df.sex_id == 3)]
    pop_df = hierarchy[['location_id', 'location_name']].merge(pop_df)
    return pop_df


def holdout_days(df: pd.DataFrame, n_holdout_days: int) -> pd.DataFrame:
    """Drop some number of holdout days from the data."""
    df = df.copy()
    df['last_date'] = df.groupby('location_id')['Date'].transform(max)
    keep_idx = df.apply(lambda x: x['Date'] <= x['last_date'] - pd.Timedelta(days=n_holdout_days), axis=1)
    df = df.loc[keep_idx].reset_index(drop=True)
    del df['last_date']

    return df


def enforce_monotonicity(df: pd.DataFrame, rate_var: str) -> pd.DataFrame:
    """Drop negatives and interpolate cumulative values."""
    vals = df[rate_var].values
    fill_idx = np.array([~(vals[i] >= vals[:i]).all() for i in range(vals.size)])
    df.loc[fill_idx, rate_var] = np.nan
    df[rate_var] = df[rate_var].interpolate()

    return df.loc[~df[rate_var].isnull()]


def filter_data_by_location(data: pd.DataFrame, hierarchy: pd.DataFrame,
                            measure: str) -> Tuple[pd.DataFrame, List[int]]:
    """Filter data based on location presence in a hierarchy."""
    extra_list = list(set(data['location_id']) - set(hierarchy['location_id']))
    logger.debug(f"{len(extra_list)} extra locations found in {measure} data. Dropping.")
    data = data.loc[~data['location_id'].isin(extra_list)]

    missing_list = list(set(hierarchy['location_id']) - set(data['location_id']))
    if missing_list:
        name_map = hierarchy.set_index('location_id')['location_name']
        missing = [(loc_id, name_map.loc[loc_id]) for loc_id in sorted(missing_list)]
        logger.warning(f'Locations {missing} are not present in {measure} data but exist in the '
                       f'modeling hierarchy.')

    return data, missing_list


def combine_data(case_data: pd.DataFrame,
                 hosp_data: pd.DataFrame,
                 death_data: pd.DataFrame,
                 pop_data: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Merge all data into a single model data set."""
    df = functools.reduce(lambda x, y: pd.merge(x, y, how='outer'),
                          [case_data.loc[:, ['location_id', 'Date', 'Confirmed case rate']],
                           hosp_data.loc[:, ['location_id', 'Date', 'Hospitalization rate']],
                           death_data.loc[:, ['location_id', 'Date', 'Death rate']],
                           pop_data.loc[:, ['location_id', 'population']]])
    df = hierarchy[['location_id', 'location_name']].merge(df)
    return df


def check_counts(model_data: pd.DataFrame, rate_var: str, action: str, threshold: int) -> pd.DataFrame:
    """Act on locations with fewer than some number of deaths/cases."""
    df = model_data.copy()
    df['Count'] = df[rate_var] * df['population']
    df['Count'] = df['Count'].fillna(0)
    sub_thresh = df.groupby('location_id')['Count'].transform(max) <= threshold
    logger.warning(f"Fewer than {threshold} {rate_var.lower().replace(' rate', 's')} for "
                   f"{';'.join(df.loc[sub_thresh, 'location_name'].unique())}\n")
    if action == 'fill_na':
        # keep rows, make these columns uninformative
        df.loc[sub_thresh, rate_var] = np.nan
    elif action == 'drop':
        # drop row entirely
        df = df.loc[~sub_thresh].reset_index(drop=True)
    else:
        raise ValueError('Invalid action specified.')
    del df['Count']

    return df


def filter_to_epi_threshold(hierarchy: pd.DataFrame,
                            model_data: pd.DataFrame,
                            threshold: int = 3) -> Tuple[pd.DataFrame, List[int], List[int]]:
    """Drop locations that don't have at least `n` deaths; do not use cases or hospitalizations if under `n`."""
    df = model_data.copy()
    df = check_counts(df, 'Confirmed case rate', 'fill_na', threshold)
    days_w_cases = df['Confirmed case rate'].notnull().groupby(df['location_id']).sum()
    no_cases_locs = days_w_cases[days_w_cases == 0].index.to_list()

    df = check_counts(df, 'Hospitalization rate', 'fill_na', threshold)
    days_w_hosp = df['Hospitalization rate'].notnull().groupby(df['location_id']).sum()
    no_hosp_locs = days_w_hosp[days_w_hosp == 0].index.to_list()

    #df = check_counts(df, 'Death rate', 'drop', threshold)
    #dropped_locations = set(hierarchy['location_id']).difference(df['location_id'])
    dropped_locations = set()

    if dropped_locations:
        logger.warning(f"Dropped {sorted(list(dropped_locations))} from data due to lack of cases or deaths.")

    return df, no_cases_locs, no_hosp_locs


def fill_dates(df: pd.DataFrame, interp_var: str = None) -> pd.DataFrame:
    """Forward fill data by date."""
    df = df.sort_values('Date').set_index('Date')
    df = df.asfreq('D').reset_index()
    if interp_var:
        df[interp_var] = df[interp_var].interpolate()
    df = df.fillna(method='pad')
    df['location_id'] = df['location_id'].astype(int)
    return df


def apply_parents(failed_model_locations: List[int], hierarchy: pd.DataFrame,
                  smooth_draws: pd.DataFrame, model_data: pd.DataFrame,
                  pop_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    failed_hierarchy = hierarchy.loc[hierarchy['location_id'].isin(failed_model_locations)].reset_index(drop=True)
    failed_hierarchy['parent_id'] = failed_hierarchy['path_to_top_parent'].apply(lambda x: int(x.split(',')[-2]))
    swip_swap = list(zip(failed_hierarchy['location_id'], failed_hierarchy['parent_id']))

    filled_draws = []
    for child_id, parent_id in swip_swap:
        draws = smooth_draws.loc[smooth_draws['location_id'] == parent_id]
        draws['location_id'] = child_id
        draws = draws.set_index(['location_id', 'date'])
        draws /= model_data.loc[model_data['location_id'] == parent_id, 'population'].values[0]
        draws *= pop_data.loc[pop_data['location_id'] == child_id, 'population'].item()
        filled_draws.append(draws.reset_index())
        parent_name = model_data.loc[model_data['location_id'] == parent_id, 'location_name'].values[0]
        child_name = model_data.loc[model_data['location_id'] == child_id, 'location_name'].values[0]
        model_data.loc[model_data['location_id'] == child_id, 'location_name'] = f'{child_name} (using {parent_name} model)'
    if filled_draws:
        filled_draws = pd.concat(filled_draws)
        smooth_draws = smooth_draws.append(filled_draws)

    return smooth_draws, model_data
