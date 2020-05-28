import os
import sys
import argparse
from functools import reduce
import pandas as pd
import numpy as np
import dill as pickle
from typing import List, Dict

from front_end_loader import load_locations, load_cases_deaths_pop, load_testing
from cfr_model import cfr_model
from smoother import synthesize_time_series
from pdf_merger import pdf_merger

import warnings
warnings.simplefilter('ignore')


def holdout_days(df: pd.DataFrame, n_holdout_days: int) -> pd.DataFrame:
    df = df.copy()
    df['last_date'] = df.groupby('location_id')['Date'].transform(max)
    keep_idx = df.apply(lambda x: x['Date'] <= x['last_date'] - pd.Timedelta(days=n_holdout_days), axis=1)
    df = df.loc[keep_idx].reset_index(drop=True)
    del df['last_date']
    
    return df


def check_counts(df: pd.DataFrame, rate_var: str, action: str, threshold: int = 3) -> pd.DataFrame:
    df['Count'] = df[rate_var] * df['population']
    df['Count'] = df['Count'].fillna(0)
    sub_thresh = df.groupby('location_id')['Count'].transform(max) <= threshold
    print(f"Fewer than {threshold} {rate_var.lower().replace(' rate', 's')} for "
          f"{';'.join(df.loc[sub_thresh, 'location_name'].unique())}\n")
    if action == 'fill_na':
        df.loc[sub_thresh, rate_var] = np.nan
    elif action == 'drop':
        df = df.loc[~sub_thresh].reset_index(drop=True)
    else:
        raise ValueError('Invalid action specified.')
    del df['Count']
    
    return df


def enforce_monotonicity(df: pd.DataFrame, rate_var: str) -> pd.DataFrame:
    vals = df[rate_var].values
    fill_idx = np.array([~(vals[i] >= vals[:i]).all() for i in range(vals.size)])
    df.loc[fill_idx, rate_var] = np.nan
    df[rate_var] = df[rate_var].interpolate()
    
    return df.loc[~df[rate_var].isnull()]


def main(location_set_version_id: int, inputs_version: str,
         run_label: str, n_holdout_days: int):
    # set up out dir
    out_dir = f'/ihme/covid-19/deaths/dev/{run_label}'
    if os.path.exists(out_dir):
        #raise ValueError('Directory already exists.')
        pass
    else:
        os.mkdir(out_dir)
    # set up model dir
    model_dir = f'{out_dir}/models'
    if os.path.exists(model_dir):
        #raise ValueError('Directory already exists.')
        pass
    else:
        os.mkdir(model_dir)
    # set up plot dir
    plot_dir = f'{out_dir}/plots'
    if os.path.exists(plot_dir):
        #raise ValueError('Directory already exists.')
        pass
    else:
        os.mkdir(plot_dir)
    
    # load all data we have
    loc_df = load_locations(location_set_version_id)
    case_df, death_df, pop_df = load_cases_deaths_pop(inputs_version)
    
    # drop days of data as specified
    case_df = holdout_days(case_df, n_holdout_days)
    death_df = holdout_days(death_df, n_holdout_days)
    
    # force cumulative to be monotonically increasing
    case_df = case_df.sort_values(['location_id', 'Date']).reset_index(drop=True)
    case_df = (case_df
               .groupby('location_id', as_index=False)
               .apply(lambda x: enforce_monotonicity(x, 'Confirmed case rate'))
               .reset_index(drop=True))
    death_df = death_df.sort_values(['location_id', 'Date']).reset_index(drop=True)
    death_df = (death_df
                .groupby('location_id', as_index=False)
                .apply(lambda x: enforce_monotonicity(x, 'Death rate'))
                .reset_index(drop=True))
    
    # add some poorly behaving locations to missing list
    # Assam (4843); Meghalaya (4862)
    bad_locations = [4843, 4862]
    
    # combine data
    df = reduce(lambda x, y: pd.merge(x, y, how='outer'),
                [case_df[['location_id', 'Date', 'Confirmed case rate']],
                 death_df[['location_id', 'Date', 'Death rate']],
                 pop_df[['location_id', 'population']]])
    df = loc_df[['location_id', 'location_name']].merge(df)
    estimating = ~df['location_id'].isin(bad_locations)
    df = df.loc[estimating]
    
    # don't use CFR model if < 2 cases; must have at least two deaths to run
    df = check_counts(df, 'Confirmed case rate', 'fill_na')
    days_w_cases = df['Confirmed case rate'].notnull().groupby(df['location_id']).sum()
    no_cases_locs = days_w_cases[days_w_cases == 0].index.to_list()
    df = check_counts(df, 'Death rate', 'drop')
    
    # fit model
    np.random.seed(15243)
    var_dict = {'dep_var':'Death rate',
                'spline_var':'Confirmed case rate',
                'indep_vars':[]}
    no_cases_df = df.loc[df['location_id'].isin(no_cases_locs)]
    df = (df.loc[~df['location_id'].isin(no_cases_locs)]
          .groupby('location_id', as_index=False)
          .apply(lambda x: cfr_model(
                  x, 
                  deaths_threshold=max(1,
                                       int((x['Death rate']*x['population']).max()*0.01)), 
                  daily=False, log=True, 
                  model_dir=model_dir,
                  **var_dict))
          .reset_index(drop=True))
    df = df.append(no_cases_df)

    # fit spline to output
    draw_df = (df.groupby('location_id', as_index=False)
               .apply(lambda x: synthesize_time_series(
                   x, 
                   #daily=True, log=True,
                   plot_dir=plot_dir, 
                   **var_dict))
               .reset_index(drop=True))
    
    # combine individual location plots
    pdf_merger(indir=plot_dir, outfile=f'{out_dir}/model_results.pdf')

    # save output
    df.to_csv(f'{out_dir}/model_data.csv', index=False)
    draw_df.to_csv(f'{out_dir}/model_results.csv', index=False)


if __name__ == '__main__':
    # take args
    parser = argparse.ArgumentParser()
    parser.add_argument('--location_set_version_id', help='IHME location hierarchy.', type=int)
    parser.add_argument('--inputs_version', help='Version tag for `model-inputs`.', type=str)
    parser.add_argument('--run_label', help='Version tag for model results.', type=str)
    parser.add_argument('--n_holdout_days', help='Number of days of data to drop.', type=int, default=0)
    args = parser.parse_args()
    
    # run model
    main(**vars(args))
