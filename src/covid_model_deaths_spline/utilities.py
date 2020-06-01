import multiprocessing
import os
from typing import Callable, List

import pandas as pd
import tqdm


def run_multiprocess(combiner: Callable[[int], pd.DataFrame],
                     location_ids: List[int],
                     pool_size: int = 20,
                     omp_num_threads: int = 4) -> List[pd.DataFrame]:
    """Runs a function in parallel using multiprocessing.

    This runner is meant to be applied to functions using MRTool features.
    They typically require some number of Open MP threads to function properly,
    though it's not super clear to me what they're used for.
    """
    # Make sure the pool has enough OpenMP threads to run the jobs.
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    try:
        with multiprocessing.Pool(pool_size) as p:
            data_dfs = list(tqdm.tqdm(p.imap(combiner, location_ids), total=len(location_ids)))
    finally:
        # Don' modify a user's environment permanently.
        del os.environ['OMP_NUM_THREADS']

    return data_dfs
