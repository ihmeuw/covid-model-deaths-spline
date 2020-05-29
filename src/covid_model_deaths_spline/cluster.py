from collections import Counter
from pathlib import Path
import shutil
import time
from typing import Dict, List

from covid_shared import shell_tools
from loguru import logger


PROJECT = 'proj_dq'
QUEUE = 'd.q'
F_MEM = '5.0G'
F_THREAD = '3'
H_RUNTIME = '00:30:00'
SLEEP_TIME = 10


def run_cluster_jobs(job_type: str, output_root: Path, job_args_map: Dict[int, List[str]]) -> None:
    import drmaa
    jobs = {}
    with drmaa.Session() as session:
        logger.info(f"Enqueuing {job_type} jobs...")
        for job_id, job_args in job_args_map.items():
            job_name = f'{job_type}_{job_id}'
            job = do_qsub(session, job_name, output_root, job_args)
            jobs[job_name] = (job, drmaa.JobState.UNDETERMINED)

        logger.info('Entering monitoring loop.')
        logger.info('-------------------------')
        logger.info('')

        while any([job[1] not in [drmaa.JobState.DONE, drmaa.JobState.FAILED] for job in jobs.values()]):

            statuses = Counter()
            for job_name, (job_id, status) in jobs.items():
                new_status = session.jobStatus(job_id)
                jobs[job_name] = (job_id, new_status)
                statuses[new_status] += 1
            for status, count in statuses.items():
                logger.info(f'{status:<35}: {count:>4}')
            logger.info('')
            time.sleep(SLEEP_TIME)
            logger.info('Checking status again')
            logger.info('---------------------')
            logger.info('')

    logger.info('**Done**')


def do_qsub(session, job_name: str, output_root: Path, script_args: List[str]) -> None:
    error_logs = output_root / job_name / 'error'
    output_logs = output_root / job_name / 'output'
    shell_tools.mkdir(error_logs, exists_ok=True)
    shell_tools.mkdir(output_logs, exists_ok=True)

    job_template = session.createJobTemplate()
    job_template.remoteCommand = shutil.which('python')
    job_template.outputPath = f':{output_logs}'
    job_template.errorPath = f':{error_logs}'
    job_template.args = script_args
    job_template.nativeSpecification = (f'-V '  # Export all environment variables
                                        f'-b y '  # Command is a binary (python)
                                        f'-P {PROJECT} '
                                        f'-q {QUEUE} '
                                        f'-l fmem={F_MEM} '
                                        f'-l fthread={F_THREAD} '
                                        f'-l h_rt={H_RUNTIME} '
                                        f'-N {job_name}')  # Name of the job
    job = session.runJob(job_template)
    logger.info(f'Submitted job {job}.')
    session.deleteJobTemplate(job_template)
    return job


def decode_status(job_status):
    """Decodes a UGE job status into a string for logging"""
    import drmaa
    decoder_map = {drmaa.JobState.UNDETERMINED: 'undetermined',
                   drmaa.JobState.QUEUED_ACTIVE: 'queued_active',
                   drmaa.JobState.SYSTEM_ON_HOLD: 'system_hold',
                   drmaa.JobState.USER_ON_HOLD: 'user_hold',
                   drmaa.JobState.USER_SYSTEM_ON_HOLD: 'user_system_hold',
                   drmaa.JobState.RUNNING: 'running',
                   drmaa.JobState.SYSTEM_SUSPENDED: 'system_suspended',
                   drmaa.JobState.USER_SUSPENDED: 'user_suspended',
                   drmaa.JobState.DONE: 'finished',
                   drmaa.JobState.FAILED: 'failed'}

    return decoder_map[job_status]
