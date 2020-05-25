"""Entry points for the deaths spline model."""
from pathlib import Path

import click
from covid_shared import paths, cli_tools
from loguru import logger

from covid_model_deaths_spline import runner


@click.command()
@cli_tools.pass_run_metadata()
@click.option('-i', '--inputs-version',
              type=click.Path(file_okay=False),
              help=('Which version of the temperature data to gather and format. '
                    'May be a full path or relative to the standard inputs root.'))
@click.option('-o', '--output-root',
              type=click.Path(file_okay=False),
              default=paths.SEIR_COVARIATES_RAW_OUTPUT_ROOT,
              show_default=True)
@click.option('-b', '--mark-best', 'mark_dir_as_best',
              is_flag=True,
              help='Marks the new outputs as best in addition to marking them as latest.')
@click.option('-p', '--production-tag',
              type=click.STRING,
              help='Tags this run as a production run.')
@cli_tools.add_verbose_and_with_debugger
def run_deaths(run_metadata,
               inputs_version, output_root,
               mark_dir_as_best, production_tag,
               verbose, with_debugger):
    """Run spline deaths model."""
    cli_tools.configure_logging_to_terminal(verbose)
    inputs_root = cli_tools.get_last_stage_directory(inputs_version,
                                                     last_stage_root=paths.MODEL_INPUTS_ROOT)
    run_metadata.update_from_path('model_inputs_metadata', inputs_root / paths.METADATA_FILE_NAME)

    output_root = Path(output_root).resolve()
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)
    run_metadata['output_path'] = str(run_directory)
    cli_tools.configure_logging_to_files(run_directory)

    main = cli_tools.monitor_application(runner.make_deaths, logger, with_debugger)
    app_metadata, _ = main(inputs_root, run_directory)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / paths.METADATA_FILE_NAME)

    cli_tools.make_links(app_metadata, run_directory, mark_dir_as_best, production_tag)

    logger.info('**Done**')
