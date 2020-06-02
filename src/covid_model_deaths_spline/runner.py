from pathlib import Path
import warnings

from covid_shared import shell_tools, cli_tools
import dill as pickle
from loguru import logger
import pandas as pd
import yaml

from covid_model_deaths_spline import data, models, pdf_merger, cluster

warnings.simplefilter('ignore')


def make_deaths(app_metadata: cli_tools.Metadata, input_root: Path, output_root: Path,
                holdout_days: int):
    #
    logger.debug("Setting up output directories.")
    model_dir = output_root / 'models'
    plot_dir = output_root / 'plots'
    shell_tools.mkdir(model_dir)
    shell_tools.mkdir(plot_dir)

    #
    logger.debug("Loading and cleaning data.")
    hierarchy = data.load_most_detailed_locations(input_root)
    full_data = data.load_full_data(input_root)
    full_data = full_data.loc[full_data['location_id'] != 60363]
    case_data = data.get_shifted_data(full_data, 'Confirmed', 'Confirmed case rate')
    hosp_data = data.get_shifted_data(full_data, 'Hospitalizations', 'Hospitalization rate')
    death_data = data.get_death_data(full_data)
    pop_data = data.get_population_data(full_data)

    #
    logger.debug(f"Dropping {holdout_days} days from the end of the data.")
    case_data = data.holdout_days(case_data, holdout_days)
    hosp_data = data.holdout_days(hosp_data, holdout_days)
    death_data = data.holdout_days(death_data, holdout_days)

    #
    logger.debug(f"Filtering data by location.")
    case_data, missing_cases = data.filter_data_by_location(case_data, hierarchy, 'cases')
    hosp_data, missing_hosp = data.filter_data_by_location(hosp_data, hierarchy, 'hospitalizations')
    death_data, missing_deaths = data.filter_data_by_location(death_data, hierarchy, 'deaths')
    pop_data, missing_pop = data.filter_data_by_location(pop_data, hierarchy, 'population')
    model_data = data.combine_data(case_data, hosp_data, death_data, pop_data, hierarchy)
    model_data, no_cases_locs, no_hosp_locs = data.filter_to_epi_threshold(model_data)

    #
    logger.debug('Preparing model settings.')
    model_settings = {}
    s1_settings = {'dep_var': 'Death rate',
                   'model_dir': str(model_dir),
                   'indep_vars': [],
                   'daily': False,
                   'log': True}
    cfr_settings = {'spline_var': 'Confirmed case rate',
                    'model_type': 'CFR'}
    cfr_settings.update(s1_settings)
    model_settings.update({'CFR': cfr_settings})
    hfr_settings = {'spline_var': 'Hospitalization rate',
                    'model_type': 'HFR'}
    hfr_settings.update(s1_settings)
    model_settings.update({'HFR': hfr_settings})
    smoother_settings = {'obs_var': 'Death rate',
                         'pred_vars': ['Predicted death rate (CFR)', 'Predicted death rate (HFR)'],
                         'spline_vars': ['Confirmed case rate', 'Hospitalization rate'],
                         'plot_dir': str(plot_dir)}
    model_settings.update({'smoother':smoother_settings})
    model_settings['no_cases_locs'] = no_cases_locs
    model_settings['no_hosp_locs'] = no_hosp_locs

    #
    logger.debug('Launching models by location.')
    working_dir = output_root / 'model_working_dir'
    shell_tools.mkdir(working_dir)
    data_path = Path(working_dir) / 'model_data.pkl'
    with data_path.open('wb') as data_file:
        pickle.dump(model_data, data_file, -1)
    results_path = Path(working_dir) / 'model_outputs'
    shell_tools.mkdir(results_path)
    model_settings['results_dir'] = str(results_path)
    settings_path = Path(working_dir) / 'settings.yaml'
    with settings_path.open('w') as settings_file:
        yaml.dump(model_settings, settings_file)
    job_args_map = {
        location_id: [models.__file__, location_id, data_path, settings_path, cluster.OMP_NUM_THREADS]
        for location_id in model_data['location_id'].unique()
    }
    cluster.run_cluster_jobs('covid_death_models', output_root, job_args_map)
    
    #
    logger.debug('Compiling results.')
    results = []
    for result_path in results_path.iterdir():
        with result_path.open('rb') as result_file:
            results.append(pickle.load(result_file))
    model_data = pd.concat([r['model_data'] for r in results]).reset_index(drop=True)
    noisy_draws = pd.concat([r['noisy_draws'] for r in results]).reset_index(drop=True)
    smooth_draws = pd.concat([r['smooth_draws'] for r in results]).reset_index(drop=True)

    #
    logger.debug("Synthesizing plots.")
    pdf_merger.pdf_merger(indir=plot_dir, outfile=str(output_root / 'model_results.pdf'))

    #
    logger.debug("Writing output data.")
    model_data = model_data.rename(columns={'Date': 'date'}).set_index(['location_id', 'date'])
    noisy_draws = noisy_draws.set_index(['location_id', 'date'])
    noisy_draws['observed'] = model_data['Death rate'].notnull().astype(int)
    smooth_draws = smooth_draws.set_index(['location_id', 'date'])
    smooth_draws['observed'] = model_data['Death rate'].notnull().astype(int)
    model_data.rename(columns={'date': 'Date'}).reset_index().to_csv(output_root / 'model_data.csv', index=False)
    noisy_draws.reset_index().to_csv(output_root / 'model_results.csv', index=False)
    smooth_draws.reset_index().to_csv(output_root / 'model_results_refit.csv', index=False)
    