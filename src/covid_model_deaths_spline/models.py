import functools
from itertools import compress
import os
from pathlib import Path
from typing import List, Dict, Tuple
import sys
from collections import namedtuple

from covid_shared.cli_tools.logging import configure_logging_to_terminal
import dill as pickle
from loguru import logger
import numpy as np
import pandas as pd
import yaml
import statsmodels.api as sm

from covid_model_deaths_spline import cfr_model, smoother, summarize, plotter

DEATH_RESULTS = namedtuple('Results', 'model_data noisy_draws smooth_draws')
INFECTION_RESULTS = namedtuple('Results', 'infections ratios')


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
    logger.info('Dropping days from data.')
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
    logger.info('Running first stage models.')
    model_data_list = [model_data.loc[:,['location_id', 'location_name', 'Date',
                                         'Death rate', 'population']]]
    if location_id not in model_settings['no_cases_locs']:
        logger.info('Launching CFR model.')
        cfr_model_data = cfr_model.cfr_model(model_data, dow_holdout=dow_holdout, **model_settings['CFR'])
        model_data_list += [cfr_model_data.loc[:, ['location_id', 'Date',
                                                   'Confirmed case rate', 'Predicted death rate (CFR)']]]
    if location_id not in model_settings['no_hosp_locs']:
        logger.info('Launching HFR model.')
        hfr_model_data = cfr_model.cfr_model(model_data, dow_holdout=dow_holdout, **model_settings['HFR'])
        model_data_list += [hfr_model_data.loc[:, ['location_id', 'Date',
                                                   'Hospitalization rate', 'Predicted death rate (HFR)']]]

    # combine outputs
    logger.info('Combining data sets.')
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
    logger.info('Launching synthesis model.')
    noisy_draws, smooth_draws = smoother.synthesize_time_series(model_data, dow_holdout=dow_holdout,
                                                                n_draws=n_draws, **model_settings['smoother'])
    model_data['dow_holdout'] = dow_holdout

    logger.info('Model iteration complete.')
    return DEATH_RESULTS(model_data, noisy_draws, smooth_draws)


def adjust_ifr(ifr: pd.Series,
               smooth_deaths: pd.Series, 
               cases: pd.Series,
               cdr_ulim: float = 0.7) -> pd.Series:
    '''
    All metrics taken in as daily.
    Cases and pseudo-deaths are indexed on date of deaths.
    Assumes IFR is indexed on date of deaths.
    '''
    cfr = (smooth_deaths / cases).rename('cfr')

    cdr = ((1 / cfr) * ifr).rename('cdr')

    ifr_adjustment = cdr_ulim / cdr
    over_threshold = ifr_adjustment > 1
    bad_2weeks = over_threshold.sum() > 14
    if bad_2weeks:
        ifr_adjustment.loc[over_threshold] = 1
    else:
        ifr_adjustment = 1
    ifr_adjustment = ifr_adjustment.rename('ifr_adjustment')
    
    adj_ifr = pd.concat([ifr, ifr_adjustment], axis=1)
    adj_ifr['ifr_adjustment'] = adj_ifr['ifr_adjustment'].fillna(method='bfill')
    adj_ifr['adj_ifr'] = adj_ifr['ifr'] * adj_ifr['ifr_adjustment']
    if bad_2weeks:
        adj_ifr['adj_ifr'] = sm.nonparametric.lowess(adj_ifr['adj_ifr'].values, 
                                                     np.arange(len(adj_ifr)),
                                                     frac=0.15).T[1]
    
    return cfr.dropna(), adj_ifr['adj_ifr'].dropna()


def get_infections(data: pd.DataFrame, ifr: pd.Series, draw_cols: List[str]) -> pd.DataFrame:
    '''
    Take in cumulative, convert to daily, apply IFR, go back to cumulative.
    Assumes IFR is indexed on date of deaths.
    Returned infections still indexed on date of deaths.
    '''
    data = (data
            .set_index(['location_id', 'Date'])
            .sort_index()
            .loc[:, draw_cols])

    # convert to daily
    data.loc[:, draw_cols] = np.diff(data.values, axis=0, prepend=0)
    
    # apply IFR
    orig_index = data.index
    data = data.divide(ifr, axis=0).loc[orig_index]
    
    # convert back to cumulative
    data = data.cumsum()
    
    return data.reset_index()


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
        f'{plot_dir}/{location_id}_deaths.pdf'
    )


def run_models(location_id: int, 
               data_path: str, ifr_path: str, hierarchy_path: str, settings_path: str,
               dow_holdouts: int, plot_dir: str, n_draws: int):
    # set seed
    logger.info(f'Starting model for location id {location_id}')
    np.random.seed(location_id)

    logger.info('Loading model inputs.')
    with Path(data_path).open('rb') as in_file:
        model_data = pickle.load(in_file)
    model_data = model_data.loc[model_data['location_id'] == location_id].reset_index(drop=True)

    logger.info('Loading hierarchy.')
    with Path(hierarchy_path).open('rb') as in_file:
        hierarchy = pickle.load(in_file)
    is_most_detailed = location_id in hierarchy['location_id'].to_list()
    
    if is_most_detailed:
        logger.info('Loading IFR.')
        path_to_top_parent = hierarchy.loc[hierarchy['location_id'] == location_id, 'path_to_top_parent'].item()
        path_to_top_parent = list(reversed([int(l) for l in path_to_top_parent.split(',')]))
        with Path(ifr_path).open('rb') as in_file:
            ifr = pickle.load(in_file)
        for ifr_location in path_to_top_parent:
            if ifr_location in ifr['location_id'].to_list():
                ifr = (ifr
                       .loc[ifr['location_id'] == ifr_location]
                       .rename(columns={'date':'Date'})
                       .set_index(['location_id', 'Date'])
                       .sort_index()
                       .loc[:, 'ifr'])
                if ifr_location != location_id:
                    logger.info(f'Using IFR for parent (location_id: {ifr_location}).')
                break
        
    logger.info('Loading settings.')
    with Path(settings_path).open() as settings_file:
        model_settings = yaml.full_load(settings_file)

    # run models
    dow_holdouts += 1
    iteration_n_draws = [int(n_draws / dow_holdouts)] * dow_holdouts
    iteration_n_draws[0] += n_draws - np.sum(iteration_n_draws)
    dow_holdouts = np.arange(dow_holdouts)
    results = []
    for h, d in zip(dow_holdouts, iteration_n_draws):
        logger.info(f'Running model iteration for holdout {h}, draw {d}')
        results.append(model_iteration(location_id, model_data, model_settings, h, d))

    logger.info('Processing results.')
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
    logger.info('Merging noisy draws.')
    noisy_draws = functools.reduce(lambda x, y: pd.merge(x, y, how='outer'), noisy_draws)
    logger.info('Merging smooth draws')
    smooth_draws = functools.reduce(lambda x, y: pd.merge(x, y, how='outer'), smooth_draws)

    # convert to infections (adjust and apply IFR in daily space)
    if is_most_detailed:
        draw_cols = [f'draw_{d}' for d in range(np.sum(iteration_n_draws))]
        cfr, adj_ifr = adjust_ifr(ifr=ifr.copy(),
                                  smooth_deaths=(smooth_draws
                                                 .set_index(['location_id', 'Date'])
                                                 .sort_index()
                                                 .loc[:, draw_cols]
                                                 .mean(axis=1)
                                                 .rename('smooth_deaths')
                                                 .diff()
                                                 .dropna()),
                                  # pseudo_deaths=(model_data
                                  #              .set_index(['location_id', 'Date'])
                                  #              .sort_index()
                                  #              .loc[:, 'Predicted death rate (CFR)']
                                  #              .rename('pseudo_deaths')
                                  #              .diff()
                                  #              .dropna()),
                                  cases=(model_data
                                         .set_index(['location_id', 'Date'])
                                         .sort_index()
                                         .loc[:, 'Confirmed case rate']
                                         .rename('cases')
                                         .diff()
                                         .dropna()))
        infections = get_infections(smooth_draws.copy(), adj_ifr, draw_cols)
        infections = infections.rename(index=str, columns={'Date':'date'})
        ratios = pd.concat([ifr, adj_ifr, cfr], axis=1).reset_index()
        ratios = ratios.rename(index=str, columns={'Date':'date'})
        plotter.infection_plots(infections.copy(),
                                model_data.copy(),
                                ratios.copy(),
                                draw_cols,
                                f'{plot_dir}/{location_id}_infections.pdf')
    
    # plot
    logger.info('Producing plots.')
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
    logger.info('Saving results.')
    output_dir = Path(model_settings['results_dir'])
    result = DEATH_RESULTS(model_data, noisy_draws, smooth_draws)
    with (output_dir / f'{location_id}_deaths.pkl').open('wb') as outfile:
        pickle.dump(result, outfile, -1)
    if is_most_detailed:
        infection_result = INFECTION_RESULTS(infections, ratios)
        with (output_dir / f'{location_id}_infections.pkl').open('wb') as outfile:
            pickle.dump(infection_result, outfile, -1)

    logger.info('**Done**')
    
    
    f'{plot_dir}/{location_id}_infections.pdf'
    
if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = sys.argv[9]
    configure_logging_to_terminal(verbose=2)  # Make the logs noisy.

    run_models(location_id=int(sys.argv[1]),
               data_path=sys.argv[2],
               ifr_path=sys.argv[3],
               hierarchy_path=sys.argv[4],
               settings_path=sys.argv[5],
               dow_holdouts=int(sys.argv[6]),
               plot_dir=sys.argv[7],
               n_draws=int(sys.argv[8]))
