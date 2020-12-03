from pathlib import Path
import warnings

from covid_shared import shell_tools, cli_tools
import dill as pickle
from loguru import logger
import pandas as pd
import yaml

from covid_model_deaths_spline import data, models, pdf_merger, cluster, summarize, aggregate

warnings.simplefilter('ignore')

PARENT_MODEL_LOCATIONS = [189]  # Tanzania


def make_deaths(app_metadata: cli_tools.Metadata, input_root: Path, output_root: Path,
                holdout_days: int, dow_holdouts: int, n_draws: int):
    logger.debug("Setting up output directories.")
    model_dir = output_root / 'models'
    spline_settings_dir = output_root / 'spline_settings'
    plot_dir = output_root / 'plots'
    shell_tools.mkdir(model_dir)
    shell_tools.mkdir(spline_settings_dir)
    shell_tools.mkdir(plot_dir)

    logger.debug("Loading and cleaning data.")
    hierarchy = data.load_most_detailed_locations(input_root)
    agg_hierarchy = data.load_aggregate_locations(input_root)

    full_data = data.load_full_data(input_root)
    full_data, manipulation_metadata = data.evil_doings(full_data)
    app_metadata.update({'data_manipulation': manipulation_metadata})

    death_data = data.get_death_data(full_data)
    max_death_date = (death_data
                      .groupby('location_id')['Date'].max()
                      .rename('max_death_date')
                      .reset_index())
    case_data = data.get_shifted_data(full_data, 'Confirmed', 'Confirmed case rate')
    case_data = case_data.merge(max_death_date)
    case_data = case_data.loc[case_data['True date'] <= case_data['max_death_date']]
    del case_data['max_death_date']
    hosp_data = data.get_shifted_data(full_data, 'Hospitalizations', 'Hospitalization rate')
    hosp_data = hosp_data.merge(max_death_date)
    hosp_data = hosp_data.loc[hosp_data['True date'] <= hosp_data['max_death_date']]
    del hosp_data['max_death_date']
    del max_death_date
    pop_data = data.get_population_data(input_root, hierarchy)

    logger.debug(f"Dropping {holdout_days} days from the end of the data.")
    case_data = data.holdout_days(case_data, holdout_days)
    hosp_data = data.holdout_days(hosp_data, holdout_days)
    death_data = data.holdout_days(death_data, holdout_days)

    logger.debug("Filtering data by location.")
    case_data, missing_cases = data.filter_data_by_location(case_data, hierarchy, 'cases')
    hosp_data, missing_hosp = data.filter_data_by_location(hosp_data, hierarchy, 'hospitalizations')
    death_data, missing_deaths = data.filter_data_by_location(death_data, hierarchy, 'deaths')
    pop_data, missing_pop = data.filter_data_by_location(pop_data, hierarchy, 'population')

    logger.debug("Combine datasets.")
    model_data = data.combine_data(case_data, hosp_data, death_data, pop_data, hierarchy)
    model_data = model_data.sort_values(['location_id', 'Date']).reset_index(drop=True)
    model_data = data.drop_leading_zeros(model_data,
                                         ['Death rate', 'Confirmed case rate', 'Hospitalization rate'])

    logger.debug("Create aggregates for modeling.")
    agg_locations = [aggregate.Location(lid, lname) for lid, lname in
                     zip(agg_hierarchy['location_id'], agg_hierarchy['location_name'])]
    agg_model_data = aggregate.compute_location_aggregates_data(
        model_data, hierarchy, agg_locations,
        ['Confirmed case rate', 'Hospitalization rate', 'Death rate']
    )
    model_data = model_data.append(agg_model_data)
    model_data = model_data.sort_values(['location_id', 'Date']).reset_index(drop=True)

    logger.debug("Filter cases/hospitalizations based on threshold.")
    model_data, dropped_locations, no_cases_locs, no_hosp_locs = data.filter_to_epi_threshold(
        hierarchy, model_data, death_threshold=5, epi_threshold=10
    )
    app_metadata.update({'dropped_locations': dropped_locations})

    logger.debug("Preparing model settings.")
    model_settings = {}
    s1_settings = {'dep_var': 'Death rate',
                   'model_dir': str(model_dir),
                   'indep_vars': []}
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
                         'spline_settings_dir': str(spline_settings_dir)}
    model_settings.update({'smoother':smoother_settings})
    model_settings['no_cases_locs'] = no_cases_locs
    model_settings['no_hosp_locs'] = no_hosp_locs

    logger.debug("Launching models by location.")
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
        location_id: [models.__file__,
                      location_id, data_path, settings_path, dow_holdouts, str(plot_dir), n_draws,
                      cluster.OMP_NUM_THREADS]
        for location_id in model_data['location_id'].unique() if location_id not in PARENT_MODEL_LOCATIONS
    }
    cluster.run_cluster_jobs('covid_death_models', output_root, job_args_map)

    logger.debug("Compiling results.")
    results = []
    for result_path in results_path.iterdir():
        with result_path.open('rb') as result_file:
            results.append(pickle.load(result_file))
    post_model_data = pd.concat([r.model_data for r in results]).reset_index(drop=True)
    noisy_draws = pd.concat([r.noisy_draws for r in results]).reset_index(drop=True)
    smooth_draws = pd.concat([r.smooth_draws for r in results]).reset_index(drop=True)
    failed_model_locations = (model_data
                              .loc[~model_data['location_id'].isin(post_model_data['location_id'].to_list()),
                                   'location_id']
                              .unique().tolist())
    failed_model_locations = [l for l in failed_model_locations if l not in PARENT_MODEL_LOCATIONS]
    failed_model_locations = [l for l in failed_model_locations if l in hierarchy['location_id'].to_list()]
    app_metadata.update({'failed_model_locations': failed_model_locations})
    model_data = post_model_data.append(model_data.loc[model_data['location_id'].isin(PARENT_MODEL_LOCATIONS)])
    obs_var = smoother_settings['obs_var']
    spline_vars = smoother_settings['spline_vars']
    
    logger.debug("Capturing location-dates with NaNs and dropping them.")
    nan_rows = smooth_draws.isnull().any(axis=1)
    smooth_draws_nans = smooth_draws.loc[nan_rows].reset_index(drop=True)
    smooth_draws = smooth_draws.loc[~nan_rows].reset_index(drop=True)
    nan_min = smooth_draws_nans.groupby('location_id')['date'].min()
    val_max = smooth_draws.groupby('location_id')['date'].max()
    date_diffs = (nan_min - val_max).apply(lambda x: x.days)
    date_diffs = date_diffs.loc[date_diffs.notnull()]
    app_metadata.update({'nan_locations': date_diffs.index.to_list()})
    if (date_diffs < 0).any():
        date_diffs.to_csv(output_root / 'problem_location_report.csv', index=False)
        raise ValueError('Dropping NaNs in middle of time series (see problem_location_report.csv)')

    logger.debug("Fill specified model locations with parent and plot them.")
    smooth_draws, model_data = data.apply_parents(PARENT_MODEL_LOCATIONS, hierarchy, smooth_draws,
                                                  model_data, pop_data)
    summarize.summarize_and_plot(
        smooth_draws.loc[smooth_draws['location_id'].isin(PARENT_MODEL_LOCATIONS)].rename(columns={'date': 'Date'}),
        model_data.loc[model_data['location_id'].isin(PARENT_MODEL_LOCATIONS)],
        str(plot_dir), obs_var=obs_var, spline_vars=spline_vars, pop_data=pop_data
    )
    app_metadata.update({'parent_model_locations': PARENT_MODEL_LOCATIONS})

    logger.debug("Make post-model aggregates and plot them.")
    agg_locations = [aggregate.Location(1, 'Global')] + agg_locations
    agg_model_data = aggregate.compute_location_aggregates_data(model_data, hierarchy, agg_locations)
    agg_model_data['location_id'] = -agg_model_data['location_id']
    agg_model_data['location_name'] = agg_model_data['location_name'] + ' (model aggregate)'
    agg_draw_df = aggregate.compute_location_aggregates_draws(smooth_draws.rename(columns={'date': 'Date'}),
                                                              hierarchy, agg_locations)
    agg_draw_df['location_id'] = -agg_draw_df['location_id']
    summarize.summarize_and_plot(agg_draw_df, agg_model_data, str(plot_dir), obs_var=obs_var, spline_vars=spline_vars)

    logger.debug("Compiling plots.")
    plot_hierarchy = aggregate.get_sorted_hierarchy_w_aggs(hierarchy, agg_hierarchy)
    possible_pdfs = ['-1.pdf'] + [f'{l}.pdf' for l in plot_hierarchy.location_id]
    existing_pdfs = [str(x).split('/')[-1] for x in plot_dir.iterdir() if x.is_file()]
    pdfs = [f'{plot_dir}/{pdf}' for pdf in possible_pdfs if pdf in existing_pdfs]
    pdf_merger.pdf_merger(pdfs=pdfs, outfile=str(output_root / 'model_results.pdf'))

    logger.debug("Writing output data.")
    model_data = model_data.rename(columns={'Date': 'date'}).set_index(['location_id', 'date'])
    noisy_draws = noisy_draws.set_index(['location_id', 'date'])
    noisy_draws['observed'] = model_data['Death rate'].notnull().astype(int)
    smooth_draws = smooth_draws.set_index(['location_id', 'date'])
    smooth_draws['observed'] = model_data['Death rate'].notnull().astype(int)
    model_data.rename(columns={'date': 'Date'}).reset_index().to_csv(output_root / 'model_data.csv', index=False)
    noisy_draws.reset_index().to_csv(output_root / 'model_results.csv', index=False)
    smooth_draws.reset_index().to_csv(output_root / 'model_results_refit.csv', index=False)
    smooth_draws_nans.to_csv(output_root / 'model_results_refit_nans.csv', index=False)
