import functools
import multiprocessing
from pathlib import Path
from typing import Callable, List
import sys

from covid_shared import shell_tools
import dill as pickle
import numpy as np
import pandas as pd
import tqdm
import yaml

from covid_model_deaths_spline.mr_spline import SplineFit


def cfr_death_threshold(data: pd.DataFrame) -> int:
    return max(1, int((data['Death rate'] * data['population']).max() * 0.01))


def cfr_model_parallel(data: pd.DataFrame,
                       model_dir: Path,
                       deaths_threshold: Callable[[pd.DataFrame], int] = cfr_death_threshold,
                       daily: bool = False,
                       log: bool = True,
                       **model_args) -> pd.DataFrame:
    _combiner = functools.partial(cfr_model,
                                  data=data, model_dir=model_dir,
                                  deaths_threshold=deaths_threshold,
                                  daily=daily, log=log, **model_args)
    location_ids = data['location_id'].unique().tolist()
    with multiprocessing.Pool(20) as p:
        model_data_dfs = list(tqdm.tqdm(p.imap(_combiner, location_ids), total=len(location_ids)))
    return pd.concat(model_data_dfs).reset_index(drop=True)


def cfr_model(location_id: int,
              data: pd.DataFrame,
              daily: bool,
              log: bool,
              dep_var: str, spline_var: str, indep_vars: List[str],
              model_dir: str) -> pd.DataFrame:
    # set up model
    np.random.seed(location_id)
    df = data[data.location_id == location_id]
    deaths_threshold = cfr_death_threshold(df)

    # add intercept
    df['intercept'] = 1

    # log transform, setting floor of 0.01 per population
    df = df.sort_values('Date').reset_index(drop=True)
    floor = 0.01 / df['population'].values[0]
    adj_vars = {}
    for orig_var in [dep_var, spline_var] + indep_vars:
        mod_var = f'Model {orig_var.lower()}'
        df[mod_var] = df[orig_var]
        if daily:
            start_idx = df.loc[~df[mod_var].isnull()].index.values[0]
            df[mod_var][start_idx+1:] = np.diff(df[mod_var].values[start_idx:])
        if log:
            df.loc[df[mod_var] < floor, mod_var] = floor
            df[mod_var] = np.log(df[mod_var])
        adj_vars.update({orig_var:mod_var})
    df['Model log'] = log
    df['Model daily'] = daily

    # keep what we can use to predict (subset further to fitting dataset below)
    non_na = ~df[list(adj_vars.values())[1:]].isnull().any(axis=1)
    df = df.loc[non_na].reset_index(drop=True)

    # lose NAs in deaths as well for modeling
    mod_df = df.copy()
    above_thresh = (mod_df[dep_var] * df['population']) >= deaths_threshold
    has_x = (mod_df[spline_var] * df['population']) >= 1
    non_na = ~mod_df[adj_vars[dep_var]].isnull()
    mod_df = mod_df.loc[above_thresh & has_x & non_na, ['intercept'] + list(adj_vars.values())].reset_index(drop=True)
    if len(mod_df) < 3:
        raise ValueError(f"Fewer than 3 days {deaths_threshold}+ deaths and 1+ cases in {df['location_name'][0]}")

    # run model and predict
    if (df[dep_var] * df['population']).max() > 20:
        n_i_knots = 5
    else:
        n_i_knots = 3
    spline_options={
        'spline_knots_type': 'frequency',
        'spline_degree': 3,
        'spline_r_linear':True,
        'spline_l_linear':True,
    }
    if not daily:
        spline_options.update({'prior_spline_monotonicity':'increasing'})
    mr_mod = SplineFit(
        data=mod_df,
        dep_var=adj_vars[dep_var],
        spline_var=adj_vars[spline_var],
        indep_vars=['intercept'] + list(map(adj_vars.get, indep_vars)),
        n_i_knots=n_i_knots,
        spline_options=spline_options,
        scale_se=False
    )
    mr_mod.fit_model()
    df['Predicted model death rate'] = mr_mod.predict(df)
    df['Predicted death rate'] = df['Predicted model death rate']
    if log:
        df['Predicted death rate'] = np.exp(df['Predicted death rate'])
    if daily:
        df['Predicted death rate'] = df['Predicted death rate'].cumsum()

    with open(f"{model_dir}/{df['location_id'][0]}.pkl", 'wb') as fwrite:
        pickle.dump(mr_mod, fwrite, -1)

    return df


def cfr_model_cluster(location_id: int, data_path: str, settings_path: str):
    with Path(data_path).open('rb') as in_file:
        in_data = pickle.load(in_file)

    with Path(settings_path).open() as settings_file:
        cfr_settings = yaml.full_load(settings_file)

    output_dir = Path(cfr_settings['results_dir'])
    shell_tools.mkdir(output_dir)
    result = cfr_model(location_id, in_data, **cfr_settings)
    with (output_dir / f'{location_id}.pkl').open('wb') as outfile:
        pickle.dump(result, outfile, -1)


if __name__ == '__main__':
    loc_id = int(sys.argv[1])
    data = sys.argv[2]
    settings = sys.argv[3]

    cfr_model_cluster(loc_id, data, settings)
