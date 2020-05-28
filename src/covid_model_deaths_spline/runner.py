from pathlib import Path
import warnings

from covid_shared import shell_tools, cli_tools
from loguru import logger

from covid_model_deaths_spline import data, cfr_model, smoother, pdf_merger

warnings.simplefilter('ignore')


def make_deaths(app_metadata: cli_tools.Metadata, input_root: Path, output_root: Path, holdout_days: int):
    logger.debug("Setting up output directories.")
    model_dir = output_root / 'models'
    plot_dir = output_root / 'plots'
    shell_tools.mkdir(model_dir)
    shell_tools.mkdir(plot_dir)

    logger.debug("Loading and cleaning data.")
    hierarchy = data.load_most_detailed_locations(input_root)
    full_data = data.load_full_data(input_root)

    case_data = data.get_shifted_case_data(full_data)
    death_data = data.get_death_data(full_data)
    pop_data = data.get_population_data(full_data)

    logger.debug(f"Dropping {holdout_days} days from the end of the data.")
    case_data = data.holdout_days(case_data, holdout_days)
    death_data = data.holdout_days(death_data, holdout_days)

    logger.debug(f"Filtering data by location.")
    case_data, missing_cases = data.filter_data_by_location(case_data, hierarchy, 'cases')
    death_data, missing_deaths = data.filter_data_by_location(death_data, hierarchy, 'deaths')
    pop_data, missing_pop = data.filter_data_by_location(pop_data, hierarchy, 'population')

    model_data = data.combine_data(case_data, death_data, pop_data, hierarchy)

    # # add some poorly behaving locations to missing list
    # # Assam (4843); Meghalaya (4862)
    # # TODO: Move to a config file or something.
    # missing_locations = [4843, 4862] # + missing_cases + missing_deaths + missing_pop
    # not_missing = ~model_data['location_id'].isin(missing_locations)
    # model_data = model_data.loc[not_missing]

    model_data, no_cases_locs = data.filter_to_threshold_cases_and_deaths(model_data)

    # fit model
    var_dict = {'dep_var': 'Death rate',
                'spline_var': 'Confirmed case rate',
                'indep_vars': []}
    logger.debug('Launching CFR model.')
    no_cases = model_data['location_id'].isin(no_cases_locs)
    no_cases_data = model_data.loc[no_cases]
    model_data = cfr_model.cfr_model_parallel(model_data.loc[~no_cases], model_dir, **var_dict)
    model_data = model_data.append(no_cases_df)
    
    logger.debug('Synthesizing time series.')
    draw_df = smoother.synthesize_time_series_parallel(model_data, plot_dir, **var_dict)

    logger.debug("Synthesizing plots.")
    pdf_merger.pdf_merger(indir=plot_dir, outfile=str(output_root / 'model_results.pdf'))

    model_data = model_data.rename(columns={'Date': 'date'}).set_index(['location_id', 'date'])
    draw_df = draw_df.set_index(['location_id', 'date'])
    draw_df['observed'] = model_data['Death rate'].notnull().astype(int)

    logger.debug("Writing output data.")
    model_data.rename(columns={'date': 'Date'}).reset_index().to_csv(output_root / 'model_data.csv', index=False)
    draw_df.reset_index().to_csv(output_root / 'model_results.csv', index=False)
