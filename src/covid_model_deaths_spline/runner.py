from pathlib import Path
import warnings

from covid_shared import shell_tools, cli_tools
import dill as pickle
from loguru import logger
import pandas as pd
import yaml

from covid_model_deaths_spline import aggregate, data, cfr_model, smoother, summarize, pdf_merger, cluster

warnings.simplefilter('ignore')


def make_deaths(app_metadata: cli_tools.Metadata, input_root: Path, output_root: Path,
                holdout_days: int, do_qsub: bool):
    logger.debug("Setting up output directories.")
    model_dir = output_root / 'models'
    plot_dir = output_root / 'plots'
    shell_tools.mkdir(model_dir)
    shell_tools.mkdir(plot_dir)

    logger.debug("Loading and cleaning data.")
    hierarchy = data.load_most_detailed_locations(input_root)
    full_data = data.load_full_data(input_root)

    case_data = data.get_shifted_data(full_data, 'Confirmed', 'Confirmed case rate')
    hosp_data = data.get_shifted_data(full_data, 'Hospitalizations', 'Hospitalization rate')
    death_data = data.get_death_data(full_data)
    pop_data = data.get_population_data(full_data)

    logger.debug(f"Dropping {holdout_days} days from the end of the data.")
    case_data = data.holdout_days(case_data, holdout_days)
    hosp_data = data.holdout_days(hosp_data, holdout_days)
    death_data = data.holdout_days(death_data, holdout_days)

    logger.debug(f"Filtering data by location.")
    case_data, missing_cases = data.filter_data_by_location(case_data, hierarchy, 'cases')
    hosp_data, missing_hosp = data.filter_data_by_location(hosp_data, hierarchy, 'hospitalizations')
    death_data, missing_deaths = data.filter_data_by_location(death_data, hierarchy, 'deaths')
    pop_data, missing_pop = data.filter_data_by_location(pop_data, hierarchy, 'population')
    model_data = data.combine_data(case_data, hosp_data, death_data, pop_data, hierarchy)
    model_data, no_cases_locs, no_hosp_locs = data.filter_to_epi_threshold(hierarchy, model_data)

    # fit model
    shared_settings = {'dep_var': 'Death rate',
                       'spline_var': 'Confirmed case rate',
                       'indep_vars': []}

    logger.debug('Launching CFR model.')
    cfr_settings = {'model_dir': str(model_dir),
                    'daily': False,
                    'log': True,
                    'model_type': 'CFR'}
    cfr_settings.update(shared_settings)

    no_cases = model_data['location_id'].isin(no_cases_locs)
    no_cases_data = model_data.loc[no_cases]
    if do_qsub:
        logger.debug('Submitting CFR jobs with qsubs')
        job_type = 'cfr_model'

        working_dir = output_root / 'cfr_working_dir'
        shell_tools.mkdir(working_dir)
        data_path = Path(working_dir) / 'model_data.pkl'
        cfr_input_data = model_data.loc[~no_cases]
        with data_path.open('wb') as data_file:
            pickle.dump(cfr_input_data, data_file, -1)

        results_path = Path(working_dir) / 'cfr_outputs'
        shell_tools.mkdir(results_path)
        cfr_settings['results_dir'] = str(results_path)

        settings_path = Path(working_dir) / 'settings.yaml'
        with settings_path.open('w') as settings_file:
            yaml.dump(cfr_settings, settings_file)

        job_args_map = {
            location_id: [cfr_model.__file__, location_id, data_path, settings_path]
            for location_id in cfr_input_data['location_id'].unique()
        }
        cluster.run_cluster_jobs(job_type, output_root, job_args_map)

        results = []
        for result_path in results_path.iterdir():
            with result_path.open('rb') as result_file:
                results.append(pickle.load(result_file))
        cfr_model_data = pd.concat(results)
    else:
        logger.debug('Running CFR models via multiprocessing.')
        cfr_model_data = cfr_model.cfr_model_parallel(model_data.loc[~no_cases], model_dir, 'CFR', **shared_settings)
    cfr_model_data = cfr_model_data.append(no_cases_data)

    logger.debug('Running HFR models (multiprocessing).')
    var_dict = {'dep_var': 'Death rate',
                'spline_var': 'Hospitalization rate',
                'indep_vars': []}
    no_hosp = model_data['location_id'].isin(no_hosp_locs)
    no_hosp_data = model_data.loc[no_hosp]
    hfr_model_data = cfr_model.cfr_model_parallel(model_data.loc[~no_hosp], model_dir, 'HFR', **var_dict)
    hfr_model_data = hfr_model_data.append(no_hosp_data)

    # combine CFR and HFR data
    model_data = cfr_model_data.loc[:,['location_id', 'location_name', 'Date',
                                       'Confirmed case rate', 'Death rate',
                                       'Predicted death rate (CFR)', 'population']].merge(
        hfr_model_data.loc[:,['location_id', 'location_name', 'Date',
                              'Hospitalization rate', 'Death rate',
                              'Predicted death rate (HFR)', 'population']],
        how='outer'
    )

    logger.debug('Synthesizing time series.')
    var_dict = {'dep_var': 'Death rate',
                'indep_vars': ['Confirmed case rate', 'Hospitalization rate']}
    draw_df = smoother.synthesize_time_series_parallel(model_data, plot_dir, **var_dict)

    agg_model_data = aggregate.compute_location_aggregates_data(model_data, hierarchy)
    agg_draw_df = aggregate.compute_location_aggregates_draws(draw_df.rename(columns={'date': 'Date'}), hierarchy)
    summarize.summarize_and_plot(agg_draw_df, agg_model_data, plot_dir, **var_dict)

    logger.debug("Synthesizing plots.")
    pdf_merger.pdf_merger(indir=plot_dir, outfile=str(output_root / 'model_results.pdf'))

    model_data = model_data.rename(columns={'Date': 'date'}).set_index(['location_id', 'date'])
    draw_df = draw_df.set_index(['location_id', 'date'])
    draw_df['observed'] = model_data['Death rate'].notnull().astype(int)

    logger.debug("Writing output data.")
    model_data.rename(columns={'date': 'Date'}).reset_index().to_csv(output_root / 'model_data.csv', index=False)
    draw_df.reset_index().to_csv(output_root / 'model_results.csv', index=False)
