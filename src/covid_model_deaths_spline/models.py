import os
from pathlib import Path
from typing import Callable, List
import sys

from covid_shared import shell_tools
import dill as pickle
import numpy as np
import pandas as pd
import tqdm
import yaml

from covid_model_deaths_spline import cfr_model, smoother


def run_models(location_id: int, data_path: str, settings_path: str):
    with Path(data_path).open('rb') as in_file:
        model_data = pickle.load(in_file)

    with Path(settings_path).open() as settings_file:
        model_settings = yaml.full_load(settings_file)

    output_dir = Path(cfr_settings['results_dir'])
    if location_id not in model_settings['no_cases_locs']:
        cfr_model_data = cfr_model.cfr_model(location_id, model_data, **model_settings['CFR'])
    if location_id not in model_settings['no_hosp_locs']:
        hfr_model_data = cfr_model.cfr_model(location_id, model_data, **model_settings['HFR'])
    model_data = cfr_model_data.loc[:,['location_id', 'location_name', 'Date', 
                                       'Confirmed case rate', 'Death rate', 
                                       'Predicted death rate (CFR)', 'population']].merge(
        hfr_model_data.loc[:,['location_id', 'location_name', 'Date', 
                              'Hospitalization rate', 'Death rate', 
                              'Predicted death rate (HFR)', 'population']],
        how='outer'
    )
    
    noisy_draws, smooth_draws = smoother.synthesize_time_series(location_id, model_data, **model_settings['HFR'])
    with (output_dir / f'{location_id}.pkl').open('wb') as outfile:
        pickle.dump(result, outfile, -1)


if __name__ == '__main__':
    location_id = int(sys.argv[1])
    data = sys.argv[2]
    settings = sys.argv[3]
    omp_num_threads = int(sys.argv[4])
    
    os.environ['OMP_NUM_THREADS'] = omp_num_threads

    run_models(location_id, data, settings)