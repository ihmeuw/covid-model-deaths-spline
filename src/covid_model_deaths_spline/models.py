import functools
from itertools import compress
import os
from pathlib import Path
from typing import List, Dict, Tuple
import sys
from collections import namedtuple

import dill as pickle
import numpy as np
import pandas as pd
import yaml

from covid_model_deaths_spline import cfr_model, smoother, summarize, plotter

RESULTS = namedtuple('Results', 'model_data noisy_draws smooth_draws')


def drop_days_by_indicator(data: np.array, deaths_data: np.array, dow_holdout: int):
    if dow_holdout > 0:
        indicator_drop_idx = np.argwhere(~np.isnan(data))[-dow_holdout:]
        deaths_drop_idx = np.argwhere(~np.isnan(deaths_data))[-dow_holdout:]

        # only drop cases/hospitalizations when leading indicator
        drop_idx = indicator_drop_idx[indicator_drop_idx >= deaths_drop_idx.min()]
        data[drop_idx] = np.nan

    return data


def model_iteration(location_id: int, model_data: pd.DataFrame, model_settings: Dict,
                    dow_holdout: int, n_draws: int):
    # drop days
    print(dow_holdout)
    model_data = model_data.copy()
    deaths_indicator = 'Death rate'
    indicators = ['Confirmed case rate', 'Hospitalization rate']
    for indicator in indicators + [deaths_indicator]:
        model_data[indicator] = drop_days_by_indicator(
            model_data[indicator].values.copy(), model_data[deaths_indicator].values.copy(),
            dow_holdout
        )
    model_data = model_data.loc[~model_data[indicators + [deaths_indicator]].isnull().all(axis=1)].reset_index(drop=True)

    # first stage model(s)
    model_data_list = [model_data.loc[:,['location_id', 'location_name', 'Date',
                                         'Death rate', 'population']]]
    if location_id not in model_settings['no_cases_locs']:
        cfr_model_data = cfr_model.cfr_model(model_data, dow_holdout=dow_holdout, **model_settings['CFR'])
        model_data_list += [cfr_model_data.loc[:, ['location_id', 'Date',
                                                   'Confirmed case rate', 'Predicted death rate (CFR)']]]
    if location_id not in model_settings['no_hosp_locs']:
        hfr_model_data = cfr_model.cfr_model(model_data, dow_holdout=dow_holdout, **model_settings['HFR'])
        model_data_list += [hfr_model_data.loc[:, ['location_id', 'Date',
                                                   'Hospitalization rate', 'Predicted death rate (HFR)']]]

    # combine outputs
    model_data = functools.reduce(
        lambda x, y: pd.merge(x, y, how='outer'),
        model_data_list
    )
    keep_cols = ['location_id', 'location_name', 'Date',
                 'Confirmed case rate', 'Hospitalization rate', 'Death rate',
                 'Predicted death rate (CFR)', 'Predicted death rate (HFR)', 'population']
    for col in keep_cols:
        if col not in model_data.columns:
            model_data[col] = np.nan
    model_data = model_data.loc[:, keep_cols]

    # second stage model
    noisy_draws, smooth_draws = smoother.synthesize_time_series(model_data, dow_holdout=dow_holdout,
                                                                n_draws=n_draws, **model_settings['smoother'])
    model_data['dow_holdout'] = dow_holdout

    return RESULTS(model_data, noisy_draws, smooth_draws)


def plot_ensemble(location_id: int, smooth_draws: pd.DataFrame, df: pd.DataFrame,
                  plot_dir: str, obs_var: str, spline_vars: List[str],
                  model_labels: List[str], draw_ranges: List[Tuple[int, int]]):
    # plot
    df = summarize.append_summary_statistics(smooth_draws.copy(), df.copy())
    plotter.plotter(
        df,
        [obs_var] + list(compress(spline_vars, (~df[spline_vars].isnull().all(axis=0)).to_list())),
        smooth_draws,
        model_labels, draw_ranges,
        f'{plot_dir}/{location_id}.pdf'
    )


def run_models(location_id: int, data_path: str, settings_path: str,
               dow_holdouts: int, plot_dir: str, n_draws: int):
    # set seed
    np.random.seed(location_id)

    # load inputs
    with Path(data_path).open('rb') as in_file:
        model_data = pickle.load(in_file)
    model_data = model_data.loc[model_data['location_id'] == location_id].reset_index(drop=True)

    with Path(settings_path).open() as settings_file:
        model_settings = yaml.full_load(settings_file)

    # run models
    dow_holdouts += 1
    iteration_n_draws = [int(n_draws / dow_holdouts)] * dow_holdouts
    iteration_n_draws[0] += n_draws - np.sum(iteration_n_draws)
    dow_holdouts = np.arange(dow_holdouts)
    results = [model_iteration(location_id, model_data, model_settings, h, d) for h, d in zip(dow_holdouts, iteration_n_draws)]

    # process results
    model_labels = []
    noisy_draws = []
    smooth_draws = []
    for i, result in enumerate(results):
        md = result.model_data
        model_label = md.loc[~md['Death rate'].isnull(), 'Date'].max()
        model_label = model_label.strftime('%m/%d/%Y (%A)')
        model_labels.append(model_label)

        col_add = int(np.sum(iteration_n_draws[:i]))
        cols = [f'draw_{d}' for d in range(iteration_n_draws[i])]
        new_cols = [f'draw_{d + col_add}' for d in range(iteration_n_draws[i])]

        nd = result.noisy_draws
        nd = nd.rename(index=str, columns=dict(zip(cols, new_cols)))
        noisy_draws.append(nd)
        sd = result.smooth_draws
        sd = sd.rename(index=str, columns=dict(zip(cols, new_cols)))
        smooth_draws.append(sd)
    model_data = results[0].model_data
    noisy_draws = functools.reduce(lambda x, y: pd.merge(x, y, how='outer'), noisy_draws)
    smooth_draws = functools.reduce(lambda x, y: pd.merge(x, y, how='outer'), smooth_draws)

    # plot
    draw_ranges = np.cumsum(iteration_n_draws)
    draw_ranges = np.append([0], draw_ranges)
    draw_ranges = list(zip(draw_ranges[:-1], draw_ranges[1:]))
    plot_ensemble(location_id, smooth_draws, model_data, plot_dir,
                  model_settings['smoother']['obs_var'],
                  model_settings['smoother']['spline_vars'],
                  model_labels, draw_ranges)
    draw_cols = [col for col in smooth_draws.columns if col.startswith('draw_')]
    noisy_draws = noisy_draws.rename(index=str, columns={'Date':'date'})
    smooth_draws = smooth_draws.rename(index=str, columns={'Date':'date'})
    noisy_draws[draw_cols] = noisy_draws[draw_cols] * noisy_draws[['population']].values
    smooth_draws[draw_cols] = smooth_draws[draw_cols] * smooth_draws[['population']].values
    del noisy_draws['population']
    del smooth_draws['population']

    # save
    result = RESULTS(model_data, noisy_draws, smooth_draws)
    output_dir = Path(model_settings['results_dir'])
    with (output_dir / f'{location_id}.pkl').open('wb') as outfile:
        pickle.dump(result, outfile, -1)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = sys.argv[7]

    run_models(location_id=int(sys.argv[1]),
               data_path=sys.argv[2],
               settings_path=sys.argv[3],
               dow_holdouts=int(sys.argv[4]),
               plot_dir=sys.argv[5],
               n_draws=int(sys.argv[6]))
