import functools
from pathlib import Path
from typing import List, Tuple

from loguru import logger
import pandas as pd


def load_most_detailed_locations(inputs_root: Path) -> pd.DataFrame:
    """Loads the most detailed locations in the current modeling hierarchy."""
    location_hierarchy_path = inputs_root / 'locations' / 'modeling_hierarchy.csv'
    hierarchy = pd.read_csv(location_hierarchy_path)

    most_detailed = hierarchy['most_detailed'] == 1
    keep_columns = ['location_id', 'location_name', 'path_to_top_parent']

    return hierarchy.loc[most_detailed, keep_columns]


def load_full_data(inputs_root: Path) -> pd.DataFrame:
    """Gets all death, case, and population data."""
    full_data_path = inputs_root / 'full_data.csv'
    data = pd.read_csv(full_data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['location_id'] = data['location_id'].astype(int)

    keep_columns = ['location_id', 'Date', 'Deaths', 'Confirmed', 'Death rate', 'population']
    sort_columns = ['location_id', 'Date']
    data = data.loc[:, keep_columns].sort_values(sort_columns).reset_index(drop=True)
    return data


def get_shifted_case_data(full_data: pd.DataFrame, shift_size: int = 8) -> pd.DataFrame:
    """Filter and clean case data and shift into the future."""
    case_df = full_data.loc[:, ['location_id', 'Date', 'Confirmed', 'population']]
    case_df['Confirmed case rate'] = case_df['Confirmed'] / case_df['population']
    case_df['True date'] = case_df['Date']
    case_df['Date'] = case_df['Date'].apply(lambda x: x + pd.Timedelta(days=shift_size))

    non_na = ~case_df['Confirmed case rate'].isnull()
    has_cases = case_df.groupby('location_id')['Confirmed case rate'].transform(max).astype(bool)
    keep_columns = ['location_id', 'True date', 'Date', 'Confirmed case rate']
    case_df = case_df.loc[non_na & has_cases, keep_columns].reset_index(drop=True)

    case_df = (case_df.groupby('location_id', as_index=False)
               .apply(lambda x: fill_dates(x))
               .reset_index(drop=True))

    return case_df


def get_death_data(full_data: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean death data."""
    non_na = ~full_data['Death rate'].isnull()
    has_deaths = full_data.groupby('location_id')['Death rate'].transform(max).astype(bool)
    keep_columns = ['location_id', 'Date', 'Death rate']
    death_df = full_data.loc[non_na & has_deaths, keep_columns].reset_index(drop=True)

    death_df = (death_df.groupby('location_id', as_index=False)
                .apply(lambda x: fill_dates(x))
                .reset_index(drop=True))

    return death_df


def get_population_data(full_data: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean population data."""
    pop_df = full_data[['location_id', 'population']].drop_duplicates()
    pop_df = pop_df.reset_index(drop=True)
    return pop_df


def holdout_days(df: pd.DataFrame, n_holdout_days: int) -> pd.DataFrame:
    """Drop some number of holdout days from the data."""
    df = df.copy()
    df['last_date'] = df.groupby('location_id')['Date'].transform(max)
    keep_idx = df.apply(lambda x: x['Date'] <= x['last_date'] - pd.Timedelta(days=n_holdout_days), axis=1)
    df = df.loc[keep_idx].reset_index(drop=True)
    del df['last_date']

    return df


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


def combine_data(case_data: pd.DataFrame, death_data: pd.DataFrame,
                 pop_data: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Merge all data into a single model data set."""
    df = functools.reduce(lambda x, y: pd.merge(x, y, how='outer'),
                          [case_data.loc[:, ['location_id', 'Date', 'Confirmed case rate']],
                           death_data.loc[:, ['location_id', 'Date', 'Death rate']],
                           pop_data.loc[:, ['location_id', 'population']]])
    df = hierarchy[['location_id', 'location_name']].merge(df)
    return df


def filter_to_two_cases_and_deaths(model_data: pd.DataFrame) -> pd.DataFrame:
    """Drop locations that don't have at least two cases and two deaths."""
    df = model_data.copy()
    df['Cases'] = model_data['Confirmed case rate'] * model_data['population']
    df = df.loc[df.groupby('location_id')['Cases'].transform(max) >= 2].reset_index(drop=True)
    del df['Cases']
    df['Deaths'] = df['Death rate'] * df['population']
    df = df.loc[df.groupby('location_id')['Deaths'].transform(max) >= 2].reset_index(drop=True)
    del df['Deaths']
    dropped_locations = set(model_data['location_id']).difference(df['location_id'])
    if dropped_locations:
        logger.warning(f"Dropped {list(dropped_locations)} from data due to lack of cases or deaths.")
    return df


def fill_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Forward fill data by date."""
    df = df.sort_values('Date').set_index('Date')
    df = df.asfreq('D', method='pad').reset_index()

    return df
