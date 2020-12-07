import os

from setuptools import setup, find_packages


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "covid_model_deaths_spline", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        'click',
        'covid_shared>=1.0.40',
        'dill',
        'loguru',
        'matplotlib',
        'numpy',
        'pandas',
        'seaborn',
        'tqdm',
        'pypdf2',
        'pyyaml',
        'ipython',
        'jupyter',
        'jupyterlab'
    ]

    test_requirements = [
        'pytest',
    ]

    doc_requirements = []

    internal_requirements = []

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=long_description,
        license=about['__license__'],
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            'docs': doc_requirements,
            'test': test_requirements,
            'internal': internal_requirements,
            'dev': [doc_requirements, test_requirements, internal_requirements]
        },

        entry_points={'console_scripts': [
            'run_deaths=covid_model_deaths_spline.cli:run_deaths'
        ]},
        zip_safe=False,
    )
