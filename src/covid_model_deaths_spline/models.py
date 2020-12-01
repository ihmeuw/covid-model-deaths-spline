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

from covid_model_deaths_spline import cfr_model, smoother, summarize, plotter
from covid_model_deaths_spline.utils import DURATION, CDR_ULIM, BAD_CDR_DAYS_THRESHOLD

from mrtool import MRData, LinearCovModel, MRBRT
from xspline import XSpline

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


def create_spline_instructions(ifr_data: pd.DataFrame,
                               int_width: int = 10,
                               buffer_width: int = 30) -> Tuple[float, np.array, np.array, bool]:
    # important dates on IFR curves
    breakpoint = ifr_data.loc[ifr_data['ifr'].diff() == 0, 'Date'].values[0]
    last_observed = ifr_data.loc[ifr_data['raw_adj_ifr'].notnull(), 'Date'].values[-1]

    # must start at least buffer_width + 5 days in; convert to time
    start_adj = ifr_data.loc[ifr_data['ifr_adjustment'] < 1, 'time'].min()
    start_adj -= int_width
    start_adj = max(buffer_width + 5, start_adj)
    breakpoint = (breakpoint - ifr_data['Date'].min()).days
    last_observed = (last_observed - ifr_data['Date'].min()).days
    end_adj = ifr_data.loc[ifr_data['ifr_adjustment'] < 1, 'time'].max()
    end_adj += int_width
    
    # flag if we are adjusting at the end of the time period
    tail = ifr_data.loc[ifr_data['time'].between(last_observed - 5, last_observed, inclusive=True)]
    tail_adj_count = (tail['ifr_adjustment'] < 1).sum()
    tail_adj_flag = tail_adj_count > 1

    # add one knot per 30 days between first and last adjustment date
    # (or IFR breakpoint if first adjustment date is after that / last adjustment date is before that)
    k1 = min(start_adj, breakpoint)
    k2 = max(breakpoint, end_adj)
    steps = max(1, int(np.round((k2 - k1) / int_width)))
    ks = np.linspace(k1, k2, steps+1)
    ks = np.round(ks).astype(int).tolist()
    
    start_value = ifr_data['ifr'].values[0]
    k1sub_value = ifr_data['ifr'].values[k1 - buffer_width]
    k_values = ifr_data['ifr'].values[ks].tolist()
    end_value = ifr_data['ifr'].values[-1]
    
    t_knots = np.array([0.,
                        (k1 - buffer_width) / ifr_data['time'].max()] +
                       [k / ifr_data['time'].max() for k in ks] +
                       [(k2 + buffer_width) / ifr_data['time'].max(),
                        1.])

    dmat0 = XSpline(t_knots, 3, True, True).design_mat(ifr_data['time'])[0]
    b0 = ((start_value - (k1sub_value * dmat0[0])) / dmat0[1]) - k1sub_value

    prior_beta_uniform = np.array([[b0,
                                    k_values[0]-k1sub_value] +
                                   [-np.inf] * (len(k_values) - 1) +
                                   [end_value-k1sub_value,
                                    end_value-k1sub_value],
                                  [b0] +
                                  [k_value - k1sub_value for k_value in k_values] +
                                  [end_value-k1sub_value,
                                   end_value-k1sub_value]])

    return k1sub_value, t_knots, prior_beta_uniform, tail_adj_flag


def smooth_ifr(ifr_data: pd.DataFrame) -> np.array:
    ifr_data = ifr_data.sort_index().reset_index()
    ifr_data['ifr'] = np.log(ifr_data['ifr'])
    ifr_data['raw_adj_ifr'] = np.log(ifr_data['raw_adj_ifr'])
    
    ifr_data['ifr_to_model'] = ifr_data['ifr']
    ifr_data.loc[ifr_data['raw_adj_ifr'].notnull(), 'ifr_to_model'] = ifr_data['raw_adj_ifr']
    ifr_data['obs_se'] = 0.1
    ifr_data['time'] = (ifr_data['Date'] - ifr_data['Date'].min()).dt.days
    ifr_data['study_id'] = 1
    ifr_data['intercept'] = 1
    
    ik_value, t_knots, prior_beta_uniform, tail_adj_flag = create_spline_instructions(ifr_data)
    
    # drop data in post-adjustment window if adjustment is at tail (Finland issue)
    t_k_values = (t_knots * ifr_data['time'].max()).astype(int)
    knot_days = ifr_data.loc[t_k_values.astype(int).tolist(), 'Date'].to_list()
    if tail_adj_flag:
        logger.info('Eliminating post-adjustment IFR data from adj_ifr fit.')
        t1 = t_k_values[-4]
        t2 = t_k_values[-2]
        window = ifr_data['time'].between(t1, t2, inclusive=False)
        ifr_data_to_model = ifr_data.loc[~window]
    else:
        ifr_data_to_model = ifr_data.copy()
    
    mr_data = MRData(
        df=ifr_data_to_model,
        col_obs='ifr_to_model',
        col_obs_se='obs_se',
        col_covs=['intercept', 'time'],
        col_study_id='study_id'
    )
    
    intercept = LinearCovModel(
        alt_cov='intercept',
        use_re=True,
        prior_beta_uniform=np.array([ik_value, ik_value]),
        prior_gamma_uniform=np.array([0., 0.]),
        name='intercept'
    )
    spline_model = LinearCovModel(
        alt_cov='time',
        use_re=False,
        use_spline=True,
        spline_knots_type='frequency',
        spline_degree=3,
        spline_l_linear=True,
        spline_r_linear=True,
        spline_knots=t_knots,
        #prior_spline_convexity='convex',
        prior_beta_uniform=prior_beta_uniform,
        name='time'
    )
    mr_model = MRBRT(data=mr_data,
                     cov_models=[intercept, spline_model])
    mr_model.fit_model()
    
    spline_model = mr_model.linear_cov_models[1]
    spline_model = spline_model.create_spline(mr_model.data)
    coefs = np.hstack([mr_model.beta_soln[0], mr_model.beta_soln[0] + mr_model.beta_soln[1:]])
    mat = spline_model.design_mat(ifr_data['time'])
    smooth_adj_ifr = mat.dot(coefs)
    smooth_adj_ifr = np.exp(smooth_adj_ifr)
    
    # ifr_data['ifr'] = np.exp(ifr_data['ifr'])
    # ifr_data['raw_adj_ifr'] = np.exp(ifr_data['raw_adj_ifr'])
    # ifr_data = ifr_data.set_index('Date')
    # plt.plot(ifr_data['ifr'])
    # plt.plot(ifr_data['raw_adj_ifr'])
    # plt.plot(ifr_data.index, smooth_adj_ifr)
    # for k in t_knots:
    #     plt.axvline(ifr_data.index[int((len(ifr_data) - 1) * k)], linestyle='--', alpha=0.5, color='grey')
    # plt.xticks(rotation=60)
    # plt.show()
    
    return smooth_adj_ifr, knot_days
    
    
def adjust_ifr(ifr: pd.Series,
               smooth_deaths: pd.Series, 
               cases: pd.Series) -> pd.Series:
    '''
    All metrics taken in as daily.
    Cases and pseudo-deaths are indexed on date of deaths.
    Assumes IFR is indexed on date of deaths.
    '''
    cfr = (smooth_deaths / cases).rename('cfr')
    cfr.loc[cfr <= 0] = np.nan

    cdr = ((1 / cfr) * ifr).rename('cdr')

    ifr_adjustment = CDR_ULIM / cdr
    ifr_adjustment.loc[ifr_adjustment > 1] = 1
    ifr_adjustment = ifr_adjustment.rename('ifr_adjustment')
    
    # adjust if we have enough days over threshold, or if we have any problems at tail
    over_threshold = ifr_adjustment.dropna() < 1
    meets_threshold = over_threshold.iloc[-DURATION:].any() or over_threshold.sum() > BAD_CDR_DAYS_THRESHOLD
    
    adj_ifr = pd.concat([ifr, ifr_adjustment], axis=1)
    past = adj_ifr['ifr_adjustment'].fillna(method='bfill').notnull()
    missing_adj = adj_ifr['ifr_adjustment'].isnull()
    adj_ifr.loc[past & missing_adj, 'ifr_adjustment'] = 1
    adj_ifr['raw_adj_ifr'] = adj_ifr['ifr'] * adj_ifr['ifr_adjustment']
    if meets_threshold:
        adj_ifr['adj_ifr'], knot_days = smooth_ifr(adj_ifr)
        adj_ifr.loc[adj_ifr['adj_ifr'] > adj_ifr['ifr'], 'adj_ifr'] = adj_ifr['ifr']
    else:
        adj_ifr['adj_ifr'] = adj_ifr['ifr']
        knot_days = []
    
    return cfr.dropna(), adj_ifr['raw_adj_ifr'].dropna(), adj_ifr['adj_ifr'].dropna(), knot_days


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
                ifr = ifr.loc[ifr['location_id'] == ifr_location]
                ifr['location_id'] = location_id
                ifr = (ifr
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
        cfr, raw_adj_ifr, adj_ifr, knot_days = adjust_ifr(
            ifr=ifr.copy(),
            smooth_deaths=(smooth_draws
                          .set_index(['location_id', 'Date'])
                          .sort_index()
                          .loc[:, draw_cols]
                          .mean(axis=1)
                          .rename('smooth_deaths')
                          .diff()
                          .dropna()),
            cases=(model_data
                  .set_index(['location_id', 'Date'])
                  .sort_index()
                  .loc[:, 'Confirmed case rate']
                  .rename('cases')
                  .diff()
                  .dropna())
        )
        infections = get_infections(smooth_draws.copy(), adj_ifr, draw_cols)
        infections = infections.rename(index=str, columns={'Date':'date'})
        ratios = pd.concat([ifr, raw_adj_ifr, adj_ifr, cfr], axis=1).reset_index()
        ratios = ratios.rename(index=str, columns={'Date':'date'})
        plotter.infection_plots(infections.copy(),
                                model_data.copy(),
                                ratios.copy(),
                                draw_cols,
                                knot_days,
                                f'{plot_dir}/{location_id}_infections.pdf')
        infections[draw_cols] = infections[draw_cols] * model_data['population'][0]
        infections[draw_cols] = np.diff(infections[draw_cols], prepend=0, axis=0)
    
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
